from typing import MutableSequence, Optional

from matplotlib.axes import Axes
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from ..models.light import Light
from .display import DisplayOne, DisplayMany, DisplaySettings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from numpy.typing import NDArray


class PolarizationEllipse(DisplayOne, DisplayMany):
    """Shows a 2D view of the polarization ellipse representing the polarization state.
    The display is animated.
    
    :param light: The light whose polarization state is shown
    :type light: Light
    :param settings: Optional settings that can be passed in to change the display's attributes
    :type settings: DisplaySettings, optional
    """
    
    _OMEGA = 0.5 * np.pi # angular_frequency
    _NUM_POINTS = 200 # amount of points used to graph the ellipse
    _AXIS_LIMIT = 1.25 # length of the axes
    _PERIODS = 3 # amount of periods shown in the animation

    DEFAULT_FPS = 60
    DEFAULT_TOTAL_TIME = (_PERIODS * 2 * np.pi) / _OMEGA # total time for one loop of the animation
    
    _INTENSITY_LIMIT = 10 ** -12 # to prevent division by zero
    
    def __init__(self, *, settings: Optional[DisplaySettings] = None):
        super().__init__(settings)

    def __str__(self) -> str:
        return (
            f"--- {self.__class__.__name__} ---\n"
            f"  Resolution: {self._NUM_POINTS} pts\n"
            f"  Frequency:  {self._OMEGA:.2f} rad/s\n"
            f"  Window:     ±{self._AXIS_LIMIT} normalized units\n"
            f"  Mode:       2D Projection (Ex-Ey plane)"
        )
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(settings={self.settings!r})"
        
    def display_one(self, light: Light, *, FPS:int = DEFAULT_FPS,
                    total_time: float = DEFAULT_TOTAL_TIME) -> None:
        """Displays information in the polarization ellipse.
        
        :param FPS: the frames per second of the animation
        :type FPS: int, optional
        :param total_time: the total time length of the animation
        :type total_time: float, optional
        """
        
        total_frames = int(FPS * total_time)
        
        # creating and positioning figure
        fig = super().create_fig()
        gs = fig.add_gridspec(2, 1, height_ratios=[15, 1])
        ax = fig.add_subplot(gs[0])
        
        self._plot_setup(ax)
        
        # plot static ellipse
        phase = np.linspace(0, 2*np.pi, self._NUM_POINTS)
        eh, ev = light.e
        magnitude = np.sqrt(np.abs(eh)**2 + np.abs(ev)**2)

        if magnitude > self._INTENSITY_LIMIT:
            eh = ((eh * np.exp(1j * phase))/magnitude).real
            ev = ((ev * np.exp(1j * phase))/magnitude).real
            ax.plot(eh, ev, label="Polarization Ellipse", color='blue', linewidth=2, linestyle='-')
            
            # plot arrow to show chirality
            arrow_idx = 0
            x_pos, y_pos = eh[arrow_idx], ev[arrow_idx]
            
            # find arrow direction
            dx = eh[arrow_idx + 1] - eh[arrow_idx]
            dy = ev[arrow_idx + 1] - ev[arrow_idx]
            
            # drawing arrow on ellipse
            ax.annotate('', 
                        xy=(x_pos + dx, y_pos + dy), # Tip of the arrow
                        xytext=(x_pos, y_pos),       # Base of the arrow
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color='blue', 
                            lw=2, 
                            mutation_scale=20 
                        ))
        else:
            ax.plot([], [])
        
        # plot specific point on the ellipse
        point, = ax.plot([], [], color="blue", ms=20, marker="o")
         
        # progress bar
        footer_pos = gs[1].get_position(fig)
        bar_width = 0.6
        bar_height = 0.03
        bar_x = 0.2
        bar_y = (footer_pos.height / 2)
        
        bar_bg = Rectangle((bar_x, bar_y), bar_width, bar_height, 
                        transform=fig.transFigure, color='gray', alpha=0.2)
        bar_fill = Rectangle((bar_x, bar_y), 0.0, bar_height, 
                            transform=fig.transFigure, color='blue', alpha=0.8)
        phase_text = fig.text(bar_x + bar_width + 0.02, bar_y + bar_height/2, '', 
                            transform=fig.transFigure, fontsize=10, va='center')
        
        fig.patches.extend([bar_bg, bar_fill])
        
        def update(frame: int) -> tuple[Line2D, Rectangle, Text]:
            """Function used in FuncAnimation to update graphics every animation frame.
            
            :param frame: The frame number of the specified frame in the animation
            :type frame: int
            :return: The updated point, the updated progress bar rectangle, and the updated
                updated phase text
            :rtype: Sequence[Line2D, Rectangle, Text]
            """
            
            progress = frame/total_frames
            t = progress*total_time
            current_phase = (self._OMEGA * t) % (2 * np.pi)
            
            eh, ev = light.e
            magnitude = np.sqrt(np.abs(eh)**2 + np.abs(ev)**2)
            
            if magnitude > self._INTENSITY_LIMIT:
                # update point position
                point.set_data([((eh * np.exp(1j * (self._OMEGA*t)))/magnitude).real],
                            [((ev * np.exp(1j * (self._OMEGA*t)))/magnitude).real])
            else:
                point.set_data([], [])
            
            # update progress bar
            progress_ratio = current_phase / (2 * np.pi)
            bar_fill.set_width(0.6 * progress_ratio)
            
            # update text
            phase_text.set_text(fr"$\mathbf{{\phi}} = {current_phase / (np.pi):.2f} \pi$ rad")
            
            return point, bar_fill, phase_text
        
        anim = FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=1000/FPS,
            blit=False,
            repeat=True
        )
        
        plt.show()

    def display_many(self, times: NDArray[np.float64], light_states: MutableSequence[Light]) -> None:
        """Displays a sequence of light states corresponding to a sequence of times on a polarization
        ellipse.
        
        :param times: The times that correspond to the light states
        :type times: NDArray[np.float64]
        :param light_states: The corresponding light states
        :type light_states: MutableSequence[Light]
        """
                
        # creating and positioning figure
        fig = super().create_fig()
        gs = fig.add_gridspec(2, 1, height_ratios=[15, 1])
        ax = fig.add_subplot(gs[0])
        
        self._plot_setup(ax)
    
        # initialize ellipse and arrow
        phase = np.linspace(0, 2*np.pi, self._NUM_POINTS)
        ellipse, = ax.plot([], [], label="Polarization Ellipse", color='blue', linewidth=2, linestyle='-')
        
        # drawing arrow on ellipse
        arrow = ax.annotate('', 
                        xy=(0, 0), 
                        xytext=(0, 0),
                        arrowprops=dict(
                            arrowstyle='-|>',
                            color='blue', 
                            lw=2, 
                            mutation_scale=20
                        ))
                
        # progress bar
        footer_pos = gs[1].get_position(fig)
        bar_width = 0.6
        bar_height = 0.03
        bar_x = 0.2
        bar_y = (footer_pos.height / 2)
        
        bar_bg = Rectangle((bar_x, bar_y), bar_width, bar_height, 
                        transform=fig.transFigure, color='gray', alpha=0.2)
        bar_fill = Rectangle((bar_x, bar_y), 0.0, bar_height, 
                            transform=fig.transFigure, color='blue', alpha=0.8)
        time_text = fig.text(bar_x + bar_width + 0.02, bar_y + bar_height/2, '', 
                            transform=fig.transFigure, fontsize=10, va='center')
        
        fig.patches.extend([bar_bg, bar_fill])
        
        # update function, called once per frame in FuncAnimation to update the point and progress bar
        def update(frame: int) -> tuple[Line2D, Quiver, Rectangle, Text]:
            """Function used in FuncAnimation to update graphics every animation frame.
            
            :param frame: The frame number of the specified frame in the animation
            :type frame: int
            :return: The updated point, the updated progress bar rectangle, and the updated
                updated phase text
            :rtype: Sequence[Line2D, Quiver, Rectangle, Text]
            """
            
            start_time = min(times)
            end_time = max(times)
            current_time = times[frame]
            total_time = end_time - start_time
            progress = (current_time - start_time) / total_time

            current_light_state = light_states[frame]
            eh, ev = current_light_state.e
            
            magnitude = np.sqrt(np.abs(eh)**2 + np.abs(ev)**2)

            # pretoects against division by zero
            if magnitude > self._INTENSITY_LIMIT:                
                # plot static ellipse
                eh = ((eh * np.exp(1j * phase))/magnitude).real
                ev = ((ev * np.exp(1j * phase))/magnitude).real
                ellipse.set_data(eh, ev)
                ellipse.set_alpha(1)

                # arrow
                idx = self._NUM_POINTS // 4
                x_base, y_base = eh[idx], ev[idx]
                dx = eh[idx + 1] - eh[idx]
                dy = ev[idx + 1] - ev[idx]
                
                # Update arrow positions
                arrow.set_visible(True)
                arrow.xy = (x_base + dx, y_base + dy)
                arrow.set_position((x_base, y_base))
            else:
                # hide ellipse and arrow if magnitude is zero
                ellipse.set_alpha(0)
                arrow.set_visible(False)
            
            # progress bar  
            bar_fill.set_width(0.6 * progress)
            
            # text that shows time
            time_text.set_text(fr"$\mathbf{{t}} = {current_time:.2f}$ s")
            
            return ellipse, arrow, bar_fill, time_text
        
        anim = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=(times[1] - times[0]) * 1000,
            blit=False,
            repeat=True
        )
        
        plt.show()

    def _plot_setup(self, ax: Axes) -> None:
        """Sets up the plot for the polarization ellipse
        
        :param ax: The axes the ellipse is plotted on
        :type ax: Axes
        """
        
        # set view
        ax.set_xlim([-self._AXIS_LIMIT, self._AXIS_LIMIT])
        ax.set_ylim([-self._AXIS_LIMIT, self._AXIS_LIMIT])
        ax.set_aspect('equal')
        
        # graph properties
        ax.grid(True)
        ax.set_xlabel(r"$\mathbf{E_h}$", fontsize=12)
        ax.set_ylabel(r"$\mathbf{E_v}$", fontsize=12)
        ax.set_title("Polarization Ellipse", fontweight="bold", pad=15)