from collections.abc import Sequence
from typing import Optional
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from ..models.light import Light
from .display import Display, DisplaySettings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class PolarizationView3D(Display):
    """Shows a 3D view of the horizontal polarization state,
    vertical polarization state, and the overall combined polarization state.
    The display is animated.
    
    :param light: The light whose polarization state is shown
    :type light: Light
    :param settings: Optional settings that can be passed in to change the display's attributes
    :type settings: DisplaySettings, optional
    """
    
    _AXIS_LIMIT = 1.25 # the length of the H and V axes
    _Z_LIMIT = 2 # the length of the r axis
    _AXIS_LABEL_OFFSET = 0.1 # the offset between the end of the axes and the label
    _WAVELENGTHS = 2 # the number of wavelengths shown
    _PERIODS = 3 # the number of periods shown within the animation before repeating
    _K = (2 * np.pi * _WAVELENGTHS) / _Z_LIMIT # wave number
    _OMEGA = 0.5 * np.pi # angular frequency
    _DZ = 0.01 # spacing between polling of z values
    
    DEFAULT_FPS = 24
    DEFAULT_TOTAL_TIME = (_PERIODS * 2 * np.pi ) / _OMEGA # total time for one loop of the animation

    def __init__(self, light: Light, *, settings: Optional[DisplaySettings] = None):
        super().__init__(light, settings)

    def display(self, *, FPS:int = DEFAULT_FPS, total_time: float = DEFAULT_TOTAL_TIME) -> None:
        """Displays information in the 3d polarization view.
        
        :param FPS: the frames per second of the animation
        :type FPS: int, optional
        :param total_time: the total time length of the animation
        :type total_time: float, optional
        """
        
        total_frames = int(FPS * total_time)
        z_array = np.linspace(0, self._Z_LIMIT, int(self._Z_LIMIT/self._DZ)) # array of points on z-axis

        # create figure
        fig = super().create_fig()
        gs = fig.add_gridspec(2, 1, height_ratios=[15, 1])
        ax = fig.add_subplot(gs[0], projection="3d")
        ax.view_init(elev=30, azim=-45, vertical_axis="y")

        # set view limits
        ax.set_xlim([-self._AXIS_LIMIT, self._AXIS_LIMIT])
        ax.set_ylim([-self._AXIS_LIMIT, self._AXIS_LIMIT])
        ax.set_zlim([-self._Z_LIMIT, self._Z_LIMIT])
        ax.set_box_aspect([1, 1, 1])
        
        # hiding axes and grid
        ax.grid(False)
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        # hiding pane background colour
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # hiding tick marks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # hiding axis lines
        ax.set_axis_off()

        # define custom axes
        line_points = np.array([-self._AXIS_LIMIT, self._AXIS_LIMIT])
        z_axis_points = np.array([0, self._Z_LIMIT])

        # H axis
        ax.plot(line_points, np.zeros(2), np.zeros(2),
                color="blue", linewidth=1, linestyle="-", label="V")
        ax.text(self._AXIS_LIMIT + self._AXIS_LABEL_OFFSET, 0, 0, r"$\mathbf{|H\rangle}$", color="blue",
                fontsize=14)

        # V axis
        ax.plot(np.zeros(2), line_points, np.zeros(2),
                color="red", linewidth=1, linestyle="-", label="R")
        ax.text(0, self._AXIS_LIMIT + self._AXIS_LABEL_OFFSET, 0, r"$\mathbf{|V\rangle}$", color="red",
                fontsize=14)

        # r axis
        ax.plot(np.zeros(2), np.zeros(2), z_axis_points,
                color="black", linewidth=1, linestyle="-", label="R")
        ax.text(0, 0, self._Z_LIMIT + self._AXIS_LABEL_OFFSET, "r", color="black",
                fontsize=14, fontweight="bold")

        # title
        ax.set_title("3D Polarization View")

        # plotting horizontal polarization curve, vertical polarization curve, total polarization curve
        eh_plot, = ax.plot([], [], [], 'o-', markersize=0,
                           linewidth=2, color="blue")
        ev_plot, = ax.plot([], [], [], 'o-', markersize=0,
                           linewidth=2, color="red")
        e_plot, = ax.plot([], [], [], 'o-', markersize=0,
                          linewidth=3.2, color="black")
        
        # plot the point on the total polarization curve at r = 0
        current_point_plot, = ax.plot(
            [], [], [], 'o-', markersize=12, linewidth=0, color="black")
        
        plots = [eh_plot, ev_plot, e_plot, current_point_plot]

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
        
        # update function, called once per frame in FuncAnimation to update the curve and progress bar
        def update(frame: int) -> Sequence[Line2D]:
            progress = frame / total_frames
            t = progress * total_time
            current_phase = (self._OMEGA * t) % (2 * np.pi)

            # update vertical, horizontal polarization curve
            eh_vals = ((self.light.e[0] *
                       np.exp(1j * (self._OMEGA*t - self._K*z_array)))/np.abs(self.light.e[0])).real
            ev_vals = ((self.light.e[1] *
                       np.exp(1j * (self._OMEGA*t - self._K*z_array)))/np.abs(self.light.e[1])).real

            # set plot data
            eh_plot.set_data(eh_vals, np.zeros_like(z_array))
            eh_plot.set_3d_properties(z_array)

            ev_plot.set_data(np.zeros_like(z_array), ev_vals)
            ev_plot.set_3d_properties(z_array)

            e_plot.set_data(eh_vals, ev_vals)
            e_plot.set_3d_properties(z_array)

            # update point on total polarization curve where r = 0
            current_point_plot.set_data([((self.light.e[0] * np.exp(1j * (self._OMEGA*t)))/np.abs(self.light.e[0])).real],
                                        [((self.light.e[1] * np.exp(1j * (self._OMEGA*t)))/np.abs(self.light.e[1])).real])
            current_point_plot.set_3d_properties([0])
            
            # update progress bar
            progress_ratio = current_phase / (2 * np.pi)
            bar_fill.set_width(0.6 * progress_ratio)
            
            # update text
            phase_text.set_text(fr"$\mathbf{{\phi}} = {current_phase / (np.pi):.2f} \pi$ rad")

            return plots, bar_fill, phase_text

        # create and run animation
        anim = FuncAnimation(
            fig,
            update,
            frames=total_frames,
            interval=1000/FPS,
            blit=False,
            repeat=True
        )

        plt.show()