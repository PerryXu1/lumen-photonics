import numpy as np
from src.lumen.models.light import Light
from src.lumen.models.stokes import Stokes
from src.lumen.display.polarization_ellipse import PolarizationEllipse
from src.lumen.display.poincare import Poincare
from src.lumen.display.polarization_view_3d import PolarizationView3D


stokes = Stokes(1, np.sqrt(1/3), -np.sqrt(1/3), np.sqrt(1/3))
light = Light(phase=0, stokes=stokes, frequency=100)

view = PolarizationEllipse(light=light)
view.display()