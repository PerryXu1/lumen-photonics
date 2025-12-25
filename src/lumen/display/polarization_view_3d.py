from typing import Optional
import numpy as np
from ..models.light import Light
from .display import Display, DisplaySettings
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PolarizationView3D(Display):
    DEFAULT_K = 3*np.pi
    DEFAULT_OMEGA = 2*np.pi

    def __init__(self, light: Light, *, settings: Optional[DisplaySettings] = None):
        super().__init__(light, settings)

    def display(self, *, k=DEFAULT_K, omega=DEFAULT_OMEGA):
        # TODO: figure out what to do with these. make parameters
        AXIS_LIMIT = 2
        FPS = 24
        TOTAL_TIME = 3
        TOTAL_FRAMES = FPS * TOTAL_TIME
        dz = 0.01
        z_array = np.linspace(0, 2, int(AXIS_LIMIT/dz))

        # ========== create figure ==========
        fig = super().create_fig()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=-45, vertical_axis="y")

        # ========== set view limits ==========
        ax.set_xlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax.set_ylim([-AXIS_LIMIT, AXIS_LIMIT])
        ax.set_zlim([-AXIS_LIMIT, AXIS_LIMIT])
        ax.set_box_aspect([1, 1, 1])

        # ========== define custom axes ==========
        line_points = np.array([-AXIS_LIMIT, AXIS_LIMIT])
        half_line_points = np.array([0, AXIS_LIMIT])

        ax.plot(line_points, np.zeros(2), np.zeros(2),
                color="blue", linewidth=1, linestyle="-", label="V")
        ax.text(AXIS_LIMIT + 0.1, 0, 0, "H", color="blue",
                fontsize=14, fontweight="bold")

        ax.plot(np.zeros(2), line_points, np.zeros(2),
                color="red", linewidth=1, linestyle="-", label="R")
        ax.text(0, AXIS_LIMIT + 0.1, 0, "V", color="red",
                fontsize=14, fontweight="bold")

        ax.plot(np.zeros(2), np.zeros(2), half_line_points,
                color="black", linewidth=1, linestyle="-", label="R")
        ax.text(0, 0, AXIS_LIMIT + 0.1, "r", color="black",
                fontsize=14, fontweight="bold")

        # ========== hiding original axes ==========
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

        ax.set_title("3D Polarization View")

        eh_plot, = ax.plot([], [], [], 'o-', markersize=0,
                           linewidth=1, color="blue")
        ev_plot, = ax.plot([], [], [], 'o-', markersize=0,
                           linewidth=1, color="red")
        e_plot, = ax.plot([], [], [], 'o-', markersize=0,
                          linewidth=1.6, color="black")
        current_point_plot, = ax.plot(
            [], [], [], 'o-', markersize=8, linewidth=0, color="black")
        plots = [eh_plot, ev_plot, e_plot, current_point_plot]

        def update(frame):
            t = (frame/TOTAL_FRAMES)*TOTAL_TIME

            eh_vals = (self.light.e[0] *
                       np.exp(1j * (k*z_array - omega*t))).real
            ev_vals = (self.light.e[1] *
                       np.exp(1j * (k*z_array - omega*t))).real

            eh_plot.set_data(eh_vals, np.zeros_like(z_array))
            eh_plot.set_3d_properties(z_array)

            ev_plot.set_data(np.zeros_like(z_array), ev_vals)
            ev_plot.set_3d_properties(z_array)

            e_plot.set_data(eh_vals, ev_vals)
            e_plot.set_3d_properties(z_array)

            current_point_plot.set_data([(self.light.e[0] * np.exp(1j * (-omega*t))).real],
                                        [(self.light.e[1] * np.exp(1j * (-omega*t))).real])
            current_point_plot.set_3d_properties([0])

            return plots

        # --- 3. Create and Run the Animation ---

        # FuncAnimation(figure, update_function, frames=iterable, interval=milliseconds_delay)
        anim = FuncAnimation(
            fig,
            update,
            frames=TOTAL_TIME*FPS,
            # Delay between frames in milliseconds (100ms = 10 FPS)
            interval=100,
            blit=False,  # Set to True for faster rendering, but can cause issues in 3D
            repeat=False
        )

        plt.show()
