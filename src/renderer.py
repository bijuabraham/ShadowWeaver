"""Rendering utilities for string art simulation using OpenCV."""

import cv2
import numpy as np


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """
    Convert hex color (#RRGGBB) to RGB tuple.

    Args:
        hex_color: Hex color string (e.g., "#FF0000")

    Returns:
        Tuple of (R, G, B) values (0-255)
    """
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def rgb_to_normalized(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Convert RGB (0-255) to normalized (0.0-1.0) values.

    Args:
        rgb: Tuple of (R, G, B) values (0-255)

    Returns:
        Tuple of normalized (R, G, B) values (0.0-1.0)
    """
    return tuple(c / 255.0 for c in rgb)


class StringArtRenderer:
    """Handles the visual simulation of string art with additive blending."""

    def __init__(
        self,
        canvas_size: int = 800,
        background_color: str = "#000000",
        thread_color: str = "#FFFFFF",
        thread_color_2: str | None = None,
    ):
        """
        Initialize the renderer.

        Args:
            canvas_size: Size of the square canvas in pixels
            background_color: Hex color for background (e.g., "#000000")
            thread_color: Primary thread color (e.g., "#FFFFFF")
            thread_color_2: Optional second thread color for alternating lines
        """
        self.canvas_size = canvas_size
        self.background_rgb = hex_to_rgb(background_color)
        self.thread_rgb = hex_to_rgb(thread_color)
        self.background_norm = rgb_to_normalized(self.background_rgb)
        self.thread_norm = rgb_to_normalized(self.thread_rgb)

        # Second thread color (optional)
        self.thread_rgb_2 = hex_to_rgb(thread_color_2) if thread_color_2 else None
        self.thread_norm_2 = rgb_to_normalized(self.thread_rgb_2) if thread_color_2 else None

        self.canvas: np.ndarray | None = None

    def create_blank_canvas(self) -> np.ndarray:
        """
        Create a canvas initialized to background color.

        Returns:
            Float32 RGB canvas initialized to background color
        """
        self.canvas = np.zeros(
            (self.canvas_size, self.canvas_size, 3), dtype=np.float32
        )
        # Fill with background color
        for i, c in enumerate(self.background_norm):
            self.canvas[:, :, i] = c
        return self.canvas

    def draw_line(
        self,
        p1: tuple,
        p2: tuple,
        thickness_px: float,
        opacity: float = 0.1,
        color_index: int = 0,
    ) -> None:
        """
        Draw a line with additive blending toward thread color.

        Uses cv2.line with anti-aliasing (cv2.LINE_AA).
        Additive process: each line blends toward thread color by 'opacity' amount.

        Args:
            p1: Starting point (x, y)
            p2: Ending point (x, y)
            thickness_px: Line thickness in pixels
            opacity: Amount to blend toward thread color per line (0.0 to 1.0)
            color_index: Which thread color to use (0 = primary, 1 = secondary)
        """
        if self.canvas is None:
            self.create_blank_canvas()

        # Select thread color based on color_index
        if color_index == 1 and self.thread_norm_2 is not None:
            thread_norm = self.thread_norm_2
        else:
            thread_norm = self.thread_norm

        # Create temporary mask for the line
        line_mask = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)

        # Draw line on mask with cv2.line (anti-aliased)
        cv2.line(
            line_mask,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color=1.0,
            thickness=max(1, int(round(thickness_px))),
            lineType=cv2.LINE_AA,
        )

        # Blend toward thread color where line is drawn
        # For each channel: canvas = canvas + (thread - canvas) * mask * opacity
        for i in range(3):
            diff = thread_norm[i] - self.canvas[:, :, i]
            self.canvas[:, :, i] += diff * line_mask * opacity

        # Clip to valid range
        self.canvas = np.clip(self.canvas, 0, 1)

    def get_display_image(self) -> np.ndarray:
        """
        Convert internal canvas to displayable uint8 RGB image.

        Returns:
            RGB image as uint8 (0-255)
        """
        if self.canvas is None:
            # Return background color canvas
            result = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
            result[:, :] = self.background_rgb
            return result
        # Convert from RGB to BGR for cv2, then back to RGB for display
        return (self.canvas * 255).astype(np.uint8)

    def get_canvas_copy(self) -> np.ndarray:
        """
        Get a copy of the current canvas state.

        Returns:
            Copy of the float32 RGB canvas
        """
        if self.canvas is None:
            result = np.zeros(
                (self.canvas_size, self.canvas_size, 3), dtype=np.float32
            )
            for i, c in enumerate(self.background_norm):
                result[:, :, i] = c
            return result
        return self.canvas.copy()
