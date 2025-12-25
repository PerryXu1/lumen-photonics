import numpy as np
from lumen.display.display import Display
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

class Poincare(Display):
    _RADIUS = 1.0
    _NUM_SPHERE_POINTS = 100
    _NUM_FRAME_LINES = 10
    _AXIS_SCALE_FACTOR = 1.25
    _AXIS_POINTS = np.array([0, _AXIS_SCALE_FACTOR * _RADIUS])
    _ZOOM = 0.85
    _AXIS_LABEL_OFFSET = 0.1
    
    def  __init__(self, light):
        super().__init__(light)
        
    def display(self):
        # ========== unit sphere ==========
        u = np.linspace(0, 2 * np.pi, self._NUM_SPHERE_POINTS)
        v = np.linspace(0, np.pi, self._NUM_SPHERE_POINTS)
        X = self._RADIUS * np.outer(np.cos(u), np.sin(v))
        Y = self._RADIUS * np.outer(np.sin(u), np.sin(v))
        Z = self._RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))

        # plot unit sphere
        fig = super().create_fig()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, color='c', alpha=0.4, rcount=50, ccount=50)

        # ========== latitude and longitude lines ==========
        # latitude lines
        u_fine = np.linspace(0, 2 * np.pi, self._NUM_SPHERE_POINTS) # For smooth circles
        v_fine = np.linspace(0, np.pi, self._NUM_SPHERE_POINTS)     # For smooth arcs
        for latitude in np.linspace(0, np.pi, self._NUM_FRAME_LINES):
            x = self._RADIUS * np.sin(latitude) * np.cos(u_fine)
            y = self._RADIUS * np.sin(latitude) * np.sin(u_fine)
            z = self._RADIUS * np.cos(latitude) * np.ones(self._NUM_SPHERE_POINTS)
            ax.plot(x, y, z, color="black", lw=0.75, alpha=0.75)

        # longitude lines
        for longitude in np.linspace(0, 2 * np.pi, self._NUM_FRAME_LINES):
            x = self._RADIUS * np.sin(v_fine) * np.cos(longitude)
            y = self._RADIUS * np.sin(v_fine) * np.sin(longitude)
            z = self._RADIUS * np.cos(v_fine)
            ax.plot(x, y, z, color="black", lw=0.75, alpha=0.75)
            
        # ========== axes ==========
        # define points on axes
        

        # V and H axis
        H = self._AXIS_POINTS
        V = -self._AXIS_POINTS
        Y_axis_x = np.zeros(2)
        Z_axis_x = np.zeros(2)
        ax.plot(H, Y_axis_x, Z_axis_x, color="red", linewidth=4, linestyle="-", label="H")
        ax.plot(V, Y_axis_x, Z_axis_x, color="green", linewidth=4, linestyle="-", label="V")

        # V and H labels
        ax.text(1.25*self._RADIUS + self._AXIS_LABEL_OFFSET, 0, 0, r"$|H \rangle$", color="black", fontsize=14)
        ax.text(-1.25*self._RADIUS - self._AXIS_LABEL_OFFSET, 0, 0, r"$|V \rangle$", color="black", fontsize=14)

        # A and D axis
        D = self._AXIS_POINTS
        A = -self._AXIS_POINTS
        X_axis_y = np.zeros(2)
        Z_axis_y = np.zeros(2)
        ax.plot(X_axis_y, D, Z_axis_y, color="yellow", linewidth=4, linestyle="-", label="L")
        ax.plot(X_axis_y, A, Z_axis_y, color="purple", linewidth=4, linestyle="-", label="R")
        
        # A and D labels
        ax.text(0, self._AXIS_SCALE_FACTOR*self._RADIUS + self._AXIS_LABEL_OFFSET, 0, r"$|D \rangle$", color="black", fontsize=14)
        ax.text(0, -self._AXIS_SCALE_FACTOR*self._RADIUS - self._AXIS_LABEL_OFFSET, 0, r"$|A \rangle$", color="black", fontsize=14)
        
        # R and L axis
        R = self._AXIS_POINTS
        L = -self._AXIS_POINTS
        X_axis_z = np.zeros(2)
        Y_axis_z = np.zeros(2)
        ax.plot(X_axis_z, Y_axis_z, R, color="orange", linewidth=4, linestyle="-", label="R")
        ax.plot(X_axis_z, Y_axis_z, L, color="blue", linewidth=4, linestyle="-", label="L")
        
        # R and L labels
        ax.text(0, 0, self._AXIS_SCALE_FACTOR * self._RADIUS + self._AXIS_LABEL_OFFSET, r"$|R \rangle$", color="black", fontsize=14)
        ax.text(0, 0, -self._AXIS_SCALE_FACTOR * self._RADIUS - self._AXIS_LABEL_OFFSET, r"$|L \rangle$", color="black", fontsize=14)
        
        # ========== plot light information ==========
        S0, S1, S2, S3 = self.light.stokes_vector()
        print(S0)
        print(S1)
        print(S2)
        print(S3)
        ax.scatter(S1/S0, S2/S0, S3/S0, c='black', marker='o', s=300)
        # ========== display ==========
        # ensure non-squashed appearance
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        X_center = (X.max()+X.min()) / 2
        Y_center = (Y.max()+Y.min()) / 2
        Z_center = (Z.max()+Z.min()) / 2
        ax.set_xlim(X_center - max_range/2, X_center + max_range/2)
        ax.set_ylim(Y_center - max_range/2, Y_center + max_range/2)
        ax.set_zlim(Z_center - max_range/2, Z_center + max_range/2)
        ax.set_box_aspect([1, 1, 1], zoom=self._ZOOM)

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
        plt.suptitle("Poincare Sphere\n", fontsize=20, fontweight='bold', y=0.95)
        plt.figtext(0.5, 0.88, f"Stokes Parameters: ({S0: .4f}, {S1: .4f}, {S2: .4f}, {S3: .4f})",
                    ha="center", fontsize=12)
        
        plt.show()