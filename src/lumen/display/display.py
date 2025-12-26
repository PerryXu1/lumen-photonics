from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from lumen.models.light import Light


@dataclass(frozen=True, slots=True)
class DisplaySettings:
    """Class that stores display settings, which can be optionally passed into a
    display's constructor to change its display settings.
    
    :param width: The width of the figure window
    :type width: int
    :param height: The height of the figure window
    :type height: int
    :param background_color: The color of the background, represented by a hex code
    :type background_color: str
    """
    
    # attributes of the figure
    width: int
    height: int
    background_color: str 

    def __post_init__(self):
        """Validates the data class
        """
        
        if self.width <= 0:
            raise ValueError("Parameter 'width' must be positive.")
        if self.height <= 0:
            raise ValueError("Parameter 'height' must be positive.")


class Display(ABC):
    """Abstract class that represents a visual display of light state information.
    
    :param light: The light whose information will be displayed
    :type light: Light
    :param settings: Optional settings that can be passed in to change characteristics
        of the display
    :type settings: DisplaySettings (optional)
    """
    
    # default settings for a display
    DEFAULT_WIDTH = 10
    DEFAULT_HEIGHT = 10
    DEFAULT_BACKGROUND_COLOR = "#E6E6E6"

    __slots__ = "light", "settings"

    def __init__(self, light: Light, settings: Optional[DisplaySettings] = None):        
        self.light = light
        # default settings used if no settings are passed in
        self.settings = settings if settings is not None else DisplaySettings(
            width=self.DEFAULT_WIDTH,
            height=self.DEFAULT_HEIGHT,
            background_color=self.DEFAULT_BACKGROUND_COLOR,
        )

    @abstractmethod
    def display(self) -> None:
        """Displays the light data"""
        
        pass

    def create_fig(self) -> Figure:
        """Creates a figure, which can be modified in the display method.
        Ensures consistency across displays.
        
        :return: A figure defined with default settings
        :rtype: Figure
        """
        
        return plt.figure(
            figsize=(self.settings.width, self.settings.height),
            facecolor=self.settings.background_color,
            constrained_layout=True
        )
