from collections.abc import Sequence
from typing import Optional
from matplotlib.lines import Line2D
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
    
    _K = 3*np.pi
    _OMEGA = 2*np.pi
    _DZ = 0.01
    _DEFAULT_AXIS_FACTOR = 1.25
    _Z_LIMIT = 2
    
    DEFAULT_FPS = 24
    DEFAULT_TOTAL_TIME = 3

    def __init__(self, light: Light, *, settings: Optional[DisplaySettings] = None):
        super().__init__(light, settings)

    def display(self, *, FPS:int = DEFAULT_FPS, total_time: float = DEFAULT_TOTAL_TIME) -> None:
        """Displays information in the 3d polarization view.
        
        :param FPS: the frames per second of the animation
        :type FPS: int, optional
        :param total_time: the total time length of the animation
        :type total_time: float, optional
        """
        
        AXIS_LIMIT = self._DEFAULT_AXIS_FACTOR * max(np.abs(self.light.e[0]),
                                                     np.abs(self.light.e[1])) # length of axes
        TOTAL_FRAMES = FPS * total_time
        z_array = np.linspace(0, self._Z_LIMIT, int(self._Z_LIMIT/self._DZ)) # array of points on z-axis

        # create figure
        fig = super().create_fig()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=-45, vertical_axis="y")

        # set view limits
        ax.set_xlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax.set_ylim([-AXIS_LIMIT, AXIS_LIMIT])
        ax.set_zlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax.set_box_aspect([1, 1, 1])

        # define custom axes
        line_points = np.array([-AXIS_LIMIT, AXIS_LIMIT])
        half_line_points = np.array([0, AXIS_LIMIT])

        # H axis
        ax.plot(line_points, np.zeros(2), np.zeros(2),
                color="blue", linewidth=1, linestyle="-", label="V")
        ax.text(AXIS_LIMIT + 0.1, 0, 0, r"$|H\rangle$", color="blue",
                fontsize=14, fontweight="bold")

        # V axis
        ax.plot(np.zeros(2), line_points, np.zeros(2),
                color="red", linewidth=1, linestyle="-", label="R")
        ax.text(0, AXIS_LIMIT + 0.1, 0, r"$|V\rangle$", color="red",
                fontsize=14, fontweight="bold")

        # r axis
        ax.plot(np.zeros(2), np.zeros(2), half_line_points,
                color="black", linewidth=1, linestyle="-", label="R")
        ax.text(0, 0, AXIS_LIMIT + 0.1, "r", color="black",
                fontsize=14, fontweight="bold")

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

        # title
        ax.set_title("3D Polarization View")

        # plotting horizontal polarization curve, vertical polarization curve, total polarization curve
        eh_plot, = ax.plot([], [], [], 'o-', markersize=0,
                           linewidth=1, color="blue")
        ev_plot, = ax.plot([], [], [], 'o-', markersize=0,
                           linewidth=1, color="red")
        e_plot, = ax.plot([], [], [], 'o-', markersize=0,
                          linewidth=1.6, color="black")
        
        # plot the point on the total polarization curve at r = 0
        current_point_plot, = ax.plot(
            [], [], [], 'o-', markersize=8, linewidth=0, color="black")
        
        plots = [eh_plot, ev_plot, e_plot, current_point_plot]

        # update function, called once per frame in FuncAnimation to update the curve
        def update(frame: int) -> Sequence[Line2D]:
            t = (frame/TOTAL_FRAMES)*total_time

            # update vertical, horizontal polarization curve
            eh_vals = (self.light.e[0] *
                       np.exp(1j * (self._K*z_array - self._OMEGA*t))).real
            ev_vals = (self.light.e[1] *
                       np.exp(1j * (self._K*z_array - self._OMEGA*t))).real

            # set plot data
            eh_plot.set_data(eh_vals, np.zeros_like(z_array))
            eh_plot.set_3d_properties(z_array)

            ev_plot.set_data(np.zeros_like(z_array), ev_vals)
            ev_plot.set_3d_properties(z_array)

            e_plot.set_data(eh_vals, ev_vals)
            e_plot.set_3d_properties(z_array)

            # update point on total polarization curve where r = 0
            current_point_plot.set_data([(self.light.e[0] * np.exp(1j * (-self._OMEGA*t))).real],
                                        [(self.light.e[1] * np.exp(1j * (-self._OMEGA*t))).real])
            current_point_plot.set_3d_properties([0])

            return plots

        # create and run animation
        anim = FuncAnimation(
            fig,
            update,
            frames=total_time*FPS,
            interval=100,
            blit=False,
            repeat=False
        )

        plt.show()