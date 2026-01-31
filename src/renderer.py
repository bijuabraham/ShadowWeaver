"""Rendering utilities for string art simulation using OpenCV."""

import cv2
import numpy as np


class StringArtRenderer:
    """Handles the visual simulation of string art with additive blending."""

    def __init__(self, canvas_size: int = 800):
        """
        Initialize the renderer.

        Args:
            canvas_size: Size of the square canvas in pixels
        """
        self.canvas_size = canvas_size
        self.canvas: np.ndarray | None = None

    def create_blank_canvas(self) -> np.ndarray:
        """
        Create a black canvas for additive rendering.

        Returns:
            Float32 canvas initialized to zeros
        """
        self.canvas = np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        return self.canvas

    def draw_line(
        self,
        p1: tuple,
        p2: tuple,
        thickness_px: float,
        opacity: float = 0.1,
    ) -> None:
        """
        Draw a line with additive blending (white thread on black background).

        Uses cv2.line with anti-aliasing (cv2.LINE_AA).
        Additive process: each line adds 'opacity' amount of white.

        Args:
            p1: Starting point (x, y)
            p2: Ending point (x, y)
            thickness_px: Line thickness in pixels
            opacity: Amount of white to add per line (0.0 to 1.0)
        """
        if self.canvas is None:
            self.create_blank_canvas()

        # Create temporary layer for the line
        line_layer = np.zeros_like(self.canvas)

        # Draw line with cv2.line (anti-aliased)
        cv2.line(
            line_layer,
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            color=1.0,  # White (normalized)
            thickness=max(1, int(round(thickness_px))),
            lineType=cv2.LINE_AA,
        )

        # Additive blending with opacity
        self.canvas = np.clip(self.canvas + line_layer * opacity, 0, 1)

    def get_display_image(self) -> np.ndarray:
        """
        Convert internal canvas to displayable uint8 image.

        Returns:
            Grayscale image as uint8 (0-255)
        """
        if self.canvas is None:
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.uint8)
        return (self.canvas * 255).astype(np.uint8)

    def get_canvas_copy(self) -> np.ndarray:
        """
        Get a copy of the current canvas state.

        Returns:
            Copy of the float32 canvas
        """
        if self.canvas is None:
            return np.zeros((self.canvas_size, self.canvas_size), dtype=np.float32)
        return self.canvas.copy()
