from collections.abc import Sequence
from typing import Optional

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from ..models.light import Light
from .display import Display, DisplaySettings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class PolarizationEllipse(Display):
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
    DEFAULT_TOTAL_TIME = (_PERIODS * 2 * np.pi) / _OMEGA # total time for one loop of the naimation
    
    def __init__(self, light: Light, *, settings: Optional[DisplaySettings] = None):
        super().__init__(light, settings)
        
    def display(self, *, FPS:int = DEFAULT_FPS, total_time: float = DEFAULT_TOTAL_TIME) -> None:
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
        fig.subplots_adjust(bottom=0.2)
        
        # set view
        ax.set_xlim([-self._AXIS_LIMIT, self._AXIS_LIMIT])
        ax.set_ylim([-self._AXIS_LIMIT, self._AXIS_LIMIT])
        ax.set_aspect('equal')
        
        # graph properties
        ax.grid(True)
        ax.set_xlabel(r"$\mathbf{E_h}$", fontsize=12)
        ax.set_ylabel(r"$\mathbf{E_v}$", fontsize=12)
        ax.set_title("Polarization Ellipse", fontweight="bold", pad=15)
        
        # plot static ellipse
        phase = np.linspace(0, 2*np.pi, self._NUM_POINTS)
        eh, ev = self.light.e
        eh = ((eh * np.exp(-1j * phase))/np.abs(eh)).real
        ev = ((ev * np.exp(-1j * phase))/np.abs(ev)).real
        ax.plot(eh, ev, label="Polarization Ellipse", color='blue', linewidth=2, linestyle='-')
        
        # plot arrow to show chirality
        arrow_idx = self._NUM_POINTS // 4
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
        
        # update function, called once per frame in FuncAnimation to update the point and progress bar
        def update(frame: int) -> Sequence[Line2D]:
            progress = frame/total_frames
            t = progress*total_time
            current_phase = (self._OMEGA * t) % (2 * np.pi)
            
            # update point position
            point.set_data([((self.light.e[0] * np.exp(1j * (-self._OMEGA*t)))/np.abs(self.light.e[0])).real],
                                        [((self.light.e[1] * np.exp(1j * (-self._OMEGA*t)))/np.abs(self.light.e[1])).real])
            
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