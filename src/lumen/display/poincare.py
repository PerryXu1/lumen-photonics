from typing import MutableSequence, Optional
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from .display import DisplayOne, DisplayMany, DisplaySettings
from ..models.light import Light

class Poincare(DisplayOne, DisplayMany):
    """A display that plots the Stokes parameters on a Poincare Sphere.
    
    :param settings: Optional settings that can be passed in to change the display's attributes
    :type settings: DisplaySettings (optional)
    """
    
    _RADIUS = 1.0 # radius of the Poincare sphere display
    _NUM_SPHERE_POINTS = 100 # amount of points used in sphere parameterization
    _NUM_FRAME_LINES = 10 # amount of lines used in the wire frame of the sphere display
    _AXIS_SCALE_FACTOR = 1.25 # radius of axis length to sphere radius
    _AXIS_POINTS = np.array([0, _AXIS_SCALE_FACTOR * _RADIUS]) # list of points used to graph axes
    _ZOOM = 0.85 # zoom on the 3d poincare display
    _AXIS_LABEL_OFFSET = 0.1 # offset of axes labels from the ends of the axes
    _FPS = 60
    _MIN_S0 = 10 ** -12 # to prevent division by zero
    
    def  __init__(self, *, settings: Optional[DisplaySettings] = None):
        super().__init__(settings)
        
    def __str__(self):
        return (
            f"--- {self.__class__.__name__} Display ---\n"
            f"  Coordinate:    Stokes (S1, S2, S3) Normalized\n"
            f"  Visuals:       3D Projection with Persistence Trail\n"
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(settings={self.settings!r})"
        
    def display_one(self, light: Light) -> None:
        """Displays a single light state on the Poincare sphere.
        
        :param light: The light to be displayed
        :type light: Light
        """
        fig = super().create_fig()
        ax = fig.add_subplot(111, projection='3d')
        
        self._draw_poincare_sphere(ax)        
        
        # plot light information on graph
        S0, S1, S2, S3 = light.stokes_vector()
        if S0 > self._MIN_S0:
            ax.scatter(S1/S0, S2/S0, S3/S0, c='black', marker='o', s=300)
        else:
            ax.scatter([], [], [], c='black', marker='o', s=300)
        
        # title
        plt.suptitle("Poincare Sphere\n", fontsize=20, fontweight='bold', y=0.95)
        plt.figtext(0.5, 0.88, f"Stokes Parameters: ({S0: .4f}, {S1: .4f}, {S2: .4f}, {S3: .4f})",
                    ha="center", fontsize=12)
        
        plt.show()
    
    def display_many(self, times: NDArray[np.float64], light_states: MutableSequence[Light]) -> None:
        """Displays a sequence of light states corresponding to a sequence of times on a Poincare sphere.
        
        :param times: The times that correspond to the light states
        :type times: NDArray[np.float64]
        :param light_states: The corresponding light states
        :type light_states: MutableSequence[Light]
        """
        
        fig = super().create_fig()
        gs = fig.add_gridspec(2, 1, height_ratios=[15, 1])
        ax = fig.add_subplot(gs[0], projection='3d')

        self._draw_poincare_sphere(ax)        
        
        # plot light information on graph
        point = ax.scatter([], [], [], c='black', marker='o', s=300)
        
        # title
        plt.suptitle("Poincare Sphere\n", fontsize=20, fontweight='bold', y=0.95)
        subtext = plt.figtext(0.5, 0.88, "",
                    ha="center", fontsize=12)
        
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
        
        trail, = ax.plot([], [], [], c='blue', alpha=0.5, linewidth=1.5)
        history_x, history_y, history_z = [], [], []
        
        def update(frame: int) -> tuple[PathCollection, Rectangle, Text, Text]:
            """Function used in FuncAnimation to update graphics every animation frame.
            
            :param frame: The frame number of the specified frame in the animation
            :type frame: int
            :return: The updated point, the updated progress bar rectangle, and the updated
                updated phase text
            :rtype: Sequence[PathCollection, Rectangle, Text, Text]
            """
            
            start_time = min(times)
            end_time = max(times)
            current_time = times[frame]
            total_time = end_time - start_time
            progress = (current_time - start_time) / total_time

            current_light_state = light_states[frame]
            S0, S1, S2, S3 = current_light_state.stokes_vector()
            
            # reset trail for each loop
            if frame == 0:
                history_x.clear()
                history_y.clear()
                history_z.clear()
            
            # prevent division by zero
            if S0 > self._MIN_S0:
                current_x, current_y, current_z = S1/S0, S2/S0, S3/S0
                point._offsets3d = ([current_x], [current_y], [current_z])
                
                history_x.append(current_x)
                history_y.append(current_y)
                history_z.append(current_z)
                point.set_alpha(1)
            else:
                point.set_alpha(0)
            
            # trail of previous posiitons
            trail.set_data(history_x, history_y)
            trail.set_3d_properties(history_z)
                      
            # progress bar  
            bar_fill.set_width(0.6 * progress)
            
            # text that shows time
            time_text.set_text(fr"$\mathbf{{t}} = {current_time:.2f}$ s")
            
            # subtitle
            subtext.set_text(f"Stokes Parameters: ({S0: .4f}, {S1: .4f}, {S2: .4f}, {S3: .4f})")

            return point, bar_fill, time_text, subtext

        # create and run animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(times),
            interval=(times[1] - times[0]) * 1000,
            blit=False,
            repeat=True
        )
        
        plt.show()
        
    def _draw_poincare_sphere(self, ax: Axes) -> None:
        """Draws the initial setup of the display, including hiding original axes, drawing
        new axes, and drawing the unit sphere.
        
        :param ax: The axes that the sphere is drawn on
        :type ax: Axes
        """
        
        # generate points on the unit sphere
        u = np.linspace(0, 2 * np.pi, self._NUM_SPHERE_POINTS)
        v = np.linspace(0, np.pi, self._NUM_SPHERE_POINTS)
        X = self._RADIUS * np.outer(np.cos(u), np.sin(v))
        Y = self._RADIUS * np.outer(np.sin(u), np.sin(v))
        Z = self._RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

        # plot unit sphere
        ax.plot_surface(X, Y, Z, color='c', alpha=0.4, rcount=50, ccount=50)

        # generate and plot latitude lines
        u_fine = np.linspace(0, 2 * np.pi, self._NUM_SPHERE_POINTS)
        v_fine = np.linspace(0, np.pi, self._NUM_SPHERE_POINTS)
        for latitude in np.linspace(0, np.pi, self._NUM_FRAME_LINES):
            x = self._RADIUS * np.sin(latitude) * np.cos(u_fine)
            y = self._RADIUS * np.sin(latitude) * np.sin(u_fine)
            z = self._RADIUS * np.cos(latitude) * np.ones(self._NUM_SPHERE_POINTS)
            ax.plot(x, y, z, color="black", lw=0.75, alpha=0.75)

        # generate and plot longitude lines
        for longitude in np.linspace(0, 2 * np.pi, self._NUM_FRAME_LINES):
            x = self._RADIUS * np.sin(v_fine) * np.cos(longitude)
            y = self._RADIUS * np.sin(v_fine) * np.sin(longitude)
            z = self._RADIUS * np.cos(v_fine)
            ax.plot(x, y, z, color="black", lw=0.75, alpha=0.75)

        # V and H axes
        H = self._AXIS_POINTS
        V = -self._AXIS_POINTS
        ax.plot(H, np.zeros(2), np.zeros(2), color="red", linewidth=4, linestyle="-", label="H")
        ax.plot(V, np.zeros(2), np.zeros(2), color="green", linewidth=4, linestyle="-", label="V")

        # V and H labels
        ax.text(1.25*self._RADIUS + self._AXIS_LABEL_OFFSET, 0, 0, r"$\mathbf{|H \rangle}$", color="black", fontsize=14)
        ax.text(-1.25*self._RADIUS - self._AXIS_LABEL_OFFSET, 0, 0, r"$\mathbf{|V \rangle}$", color="black", fontsize=14)

        # A and D axes
        D = self._AXIS_POINTS
        A = -self._AXIS_POINTS
        ax.plot(np.zeros(2), D, np.zeros(2), color="yellow", linewidth=4, linestyle="-", label="L")
        ax.plot(np.zeros(2), A, np.zeros(2), color="purple", linewidth=4, linestyle="-", label="R")
        
        # A and D labels
        ax.text(0, self._AXIS_SCALE_FACTOR*self._RADIUS + self._AXIS_LABEL_OFFSET, 0, r"$\mathbf{|D \rangle}$", color="black", fontsize=14)
        ax.text(0, -self._AXIS_SCALE_FACTOR*self._RADIUS - self._AXIS_LABEL_OFFSET, 0, r"$\mathbf{|A \rangle}$", color="black", fontsize=14)
        
        # R and L axes
        R = self._AXIS_POINTS
        L = -self._AXIS_POINTS
        ax.plot(np.zeros(2), np.zeros(2), R, color="orange", linewidth=4, linestyle="-", label="R")
        ax.plot(np.zeros(2), np.zeros(2), L, color="blue", linewidth=4, linestyle="-", label="L")
        
        # R and L labels
        ax.text(0, 0, self._AXIS_SCALE_FACTOR * self._RADIUS + self._AXIS_LABEL_OFFSET, r"$\mathbf{|R \rangle}$", color="black", fontsize=14)
        ax.text(0, 0, -self._AXIS_SCALE_FACTOR * self._RADIUS - self._AXIS_LABEL_OFFSET, r"$\mathbf{|L \rangle}$", color="black", fontsize=14)

        # ensure non-squashed appearance
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        X_center = (X.max()+X.min()) / 2
        Y_center = (Y.max()+Y.min()) / 2
        Z_center = (Z.max()+Z.min()) / 2
        ax.set_xlim(X_center - max_range/2, X_center + max_range/2)
        ax.set_ylim(Y_center - max_range/2, Y_center + max_range/2)
        ax.set_zlim(Z_center - max_range/2, Z_center + max_range/2)
        ax.set_box_aspect([1, 1, 1], zoom=self._ZOOM) # setting zoom

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