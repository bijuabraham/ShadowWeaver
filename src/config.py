"""Configuration constants for Lumina String Art."""


class Config:
    """Default configuration values for the string art generator."""

    # Physical defaults
    DEFAULT_NAIL_COUNT = 179
    DEFAULT_DIAMETER_INCHES = 13.75
    DEFAULT_THREAD_THICKNESS_MM = 0.25
    DEFAULT_LINE_OPACITY = 0.1

    # Algorithm defaults
    DEFAULT_TOTAL_LINES = 2000
    DEFAULT_MIN_NAIL_SKIP = 20
    DEFAULT_DECAY_FACTOR = 0.85

    # Canvas settings
    CANVAS_SIZE_PX = 800

    # Color defaults
    DEFAULT_BACKGROUND_COLOR = "#000000"
    DEFAULT_THREAD_COLOR = "#FFFFFF"

    @staticmethod
    def mm_to_pixels(mm: float, diameter_inches: float, canvas_px: int = 800) -> float:
        """
        Convert millimeters to pixels based on physical diameter.

        Args:
            mm: Thread thickness in millimeters
            diameter_inches: Physical board diameter in inches
            canvas_px: Canvas size in pixels

        Returns:
            Thread thickness in pixels (minimum 1.0)
        """
        diameter_mm = diameter_inches * 25.4
        pixels_per_mm = canvas_px / diameter_mm
        return max(1.0, mm * pixels_per_mm)
