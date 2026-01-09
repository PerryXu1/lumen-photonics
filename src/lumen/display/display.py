from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import MutableSequence, Optional
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from lumen.models.light import Light
from numpy.typing import NDArray
import numpy as np

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
    
    def __repr__(self):
        return (f"{self.__class__.__name__}(width={self.width!r}, "
                f"height={self.height!r}, background_color={self.background_color!r})")

    def __str__(self):
        return f"DisplaySettings: {self.width}x{self.height} [BG: {self.background_color}]"

    def __post_init__(self):
        """Validates the data class
        """
        
        if self.width <= 0:
            raise ValueError("Parameter 'width' must be positive.")
        if self.height <= 0:
            raise ValueError("Parameter 'height' must be positive.")


class DisplayOne(ABC):
    """Abstract class that represents a visual display of a single light state.
    
    :param settings: Optional settings that can be passed in to change characteristics
        of the display
    :type settings: DisplaySettings (optional)
    """
    
    # default settings for a display
    _DEFAULT_WIDTH = 10
    _DEFAULT_HEIGHT = 10
    _DEFAULT_BACKGROUND_COLOR = "#E6E6E6"

    def __init__(self, settings: Optional[DisplaySettings] = None):        
        # default settings used if no settings are passed in
        self.settings = settings if settings is not None else DisplaySettings(
            width=self._DEFAULT_WIDTH,
            height=self._DEFAULT_HEIGHT,
            background_color=self._DEFAULT_BACKGROUND_COLOR,
        )
        
    def __repr__(self):
        return f"{self.__class__.__name__}(settings={self.settings!r})"

    def __str__(self):
        return f"<{self.__class__.__name__}> visualizer for a single Light state ({self.settings.width}x{self.settings.height})"

    @abstractmethod
    def display_one(self, light) -> None:
        """Displays the light state."""
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

class DisplayMany(ABC):
    """Abstract class that represents a visual display of many light states.
    
    :param settings: Optional settings that can be passed in to change characteristics
        of the display
    :type settings: DisplaySettings (optional)
    """
    
    # default settings for a display
    _DEFAULT_WIDTH = 10
    _DEFAULT_HEIGHT = 10
    _DEFAULT_BACKGROUND_COLOR = "#E6E6E6"

    def __init__(self, settings: Optional[DisplaySettings] = None):        
        # default settings used if no settings are passed in
        self.settings = settings if settings is not None else DisplaySettings(
            width=self._DEFAULT_WIDTH,
            height=self._DEFAULT_HEIGHT,
            background_color=self._DEFAULT_BACKGROUND_COLOR,
        )

    def __str__(self):
        return f"<{self.__class__.__name__}> visualizer for multiple Light states ({self.settings.width}x{self.settings.height})"

    def __repr__(self):
        return f"{self.__class__.__name__}(settings={self.settings!r})"

    @abstractmethod
    def display_many(self, times: NDArray[np.float64], light_states: MutableSequence[Light]) -> None:
        """Displays the sequence of light states."""
        
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
