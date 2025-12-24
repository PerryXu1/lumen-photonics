import numpy as np
from lumen.display.display import Display
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

class Poincare(Display):
    
    def  __init__(self, light):
        super().__init__(light)
        
    def display(self):
        # ========== unit sphere ==========
        radius = 1.0
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        X = radius * np.outer(np.cos(u), np.sin(v))
        Y = radius * np.outer(np.sin(u), np.sin(v))
        Z = radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # plot unit sphere
        fig = plt.figure(figsize=(10, 10), facecolor="E6E6E6", constrained_layout=True)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z, color='c', alpha=0.6, rcount=50, ccount=50)

        # ========== latitude and longitude lines ==========
        N_lines = 10
        # latitude lines
        for latitude in np.linspace(0, np.pi, N_lines):
            x = radius * np.sin(latitude) * np.cos(u_fine)
            y = radius * np.sin(latitude) * np.sin(u_fine)
            z = radius * np.cos(latitude) * np.ones(100)
            ax.plot(x, y, z, color="black", lw=0.75, alpha=0.75)

        # longitude lines
        for longitude in np.linspace(0, 2 * np.pi, N_lines):
            x = radius * np.sin(v_fine) * np.cos(longitude)
            y = radius * np.sin(v_fine) * np.sin(longitude)
            z = radius * np.cos(v_fine)
            ax.plot(x, y, z, color="black", lw=0.75, alpha=0.75)
            
        # ========== axes ==========
        # define points on axes
        line_points = np.array([0, 1.25*radius]) 

        # V and H axis
        V = line_points
        H = -line_points
        Y_axis_x = np.zeros(2)
        Z_axis_x = np.zeros(2)
        ax.plot(V, Y_axis_x, Z_axis_x, color="green", linewidth=4, linestyle="-", label="V")
        ax.plot(H, Y_axis_x, Z_axis_x, color="red", linewidth=4, linestyle="-", label="H")

        # V and H labels
        ax.text(1.25*radius + 0.1, 0, 0, "V", color="black", fontsize=14)
        ax.text(-1.25*radius - 0.1, 0, 0, "H", color="black", fontsize=14)

        # R and L axis
        R = line_points
        L = -line_points
        X_axis_z = np.zeros(2)
        Y_axis_z = np.zeros(2)
        ax.plot(X_axis_z, Y_axis_z, R, color="orange", linewidth=4, linestyle="-", label="R")
        ax.plot(X_axis_z, Y_axis_z, L, color="blue", linewidth=4, linestyle="-", label="L")
        
        # R and L labels
        ax.text(0, 0, 1.25*radius + 0.1, "R", color="black", fontsize=14)
        ax.text(0, 0, -1.25*radius - 0.1, "L", color="black", fontsize=14)
        
        # A and D axis
        A = line_points
        D = -line_points
        X_axis_y = np.zeros(2)
        Z_axis_y = np.zeros(2)
        ax.plot(X_axis_y, A, Z_axis_y, color="purple", linewidth=4, linestyle="-", label="R")
        ax.plot(X_axis_y, D, Z_axis_y, color="yellow", linewidth=4, linestyle="-", label="L")
        
        # A and D labels
        ax.text(0, 1.25*radius + 0.1, 0, "A", color="black", fontsize=14)
        ax.text(0, -1.25*radius - 0.1, 0, "D", color="black", fontsize=14)

        # ========== plot light information ==========
        S0, S1, S2, S3 = self.light.stokes
        ax.scatter(S1/S0, S2/S0, S3/S0, c='k', marker='o', s=100)

        # ========== display ==========
        # ensure non-squashed appearance
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        X_center = (X.max()+X.min()) / 2
        Y_center = (Y.max()+Y.min()) / 2
        Z_center = (Z.max()+Z.min()) / 2
        ax.set_xlim(X_center - max_range/2, X_center + max_range/2)
        ax.set_ylim(Y_center - max_range/2, Y_center + max_range/2)
        ax.set_zlim(Z_center - max_range/2, Z_center + max_range/2)
        ax.set_box_aspect([1, 1, 1], zoom=0.85)

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
        plt.figtext(0.5, 0.88, f"Stokes Parameters: ({S0}, {S1}, {S2}, {S3})", ha="center", fontsize=12)
        plt.show()