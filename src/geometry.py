"""Geometry utilities for nail coordinate calculations."""

import numpy as np
import streamlit as st


@st.cache_data
def calculate_nail_positions(num_nails: int, canvas_size: int = 800) -> np.ndarray:
    """
    Pre-calculate all nail positions using vectorized NumPy operations.

    Formula: x = center + R*cos(theta), y = center + R*sin(theta)
    where theta = 2*pi*i/num_nails for i in [0, num_nails)

    Args:
        num_nails: Number of nails around the circular board
        canvas_size: Size of the square canvas in pixels

    Returns:
        np.ndarray of shape (num_nails, 2) with (x, y) coordinates as int32
    """
    radius = (canvas_size - 2) // 2  # Leave 1px margin
    center = canvas_size // 2

    # Vectorized angle calculation
    angles = np.linspace(0, 2 * np.pi, num_nails, endpoint=False)

    # Vectorized coordinate calculation
    x = center + radius * np.cos(angles)
    y = center + radius * np.sin(angles)

    return np.column_stack([x, y]).astype(np.int32)


def get_line_pixels(p1: tuple, p2: tuple) -> np.ndarray:
    """
    Get all pixel coordinates along a line using NumPy interpolation.

    More efficient than iterative Bresenham for our use case.
    Uses numpy.linspace for coordinate interpolation.

    Args:
        p1: Starting point (x, y)
        p2: Ending point (x, y)

    Returns:
        np.ndarray of shape (N, 2) with (x, y) coordinates
    """
    x1, y1 = p1
    x2, y2 = p2

    # Determine number of samples based on longer axis
    length = int(max(abs(x2 - x1), abs(y2 - y1)))
    if length == 0:
        return np.array([[int(x1), int(y1)]])

    # Interpolate coordinates
    t = np.linspace(0, 1, length + 1)
    x_coords = np.round(x1 + t * (x2 - x1)).astype(int)
    y_coords = np.round(y1 + t * (y2 - y1)).astype(int)

    return np.column_stack([x_coords, y_coords])


def calculate_circular_distance(nail_a: int, nail_b: int, num_nails: int) -> int:
    """
    Calculate the minimum circular distance between two nails.

    Args:
        nail_a: First nail index
        nail_b: Second nail index
        num_nails: Total number of nails

    Returns:
        Minimum distance (going either direction around the circle)
    """
    direct = abs(nail_b - nail_a)
    return min(direct, num_nails - direct)


@st.cache_data
def create_center_weight_map(
    canvas_size: int = 800,
    center_penalty: float = 0.3,
    center_radius_pct: float = 0.5,
) -> np.ndarray:
    """
    Create a weight map that penalizes center pixels to prioritize edge features.

    Pixels within center_radius_pct have center_penalty weight (lower score).
    Pixels outside have weight 1.0 (full importance for edges like jawline, hair).

    Args:
        canvas_size: Size of the square canvas in pixels
        center_penalty: Weight for center pixels (0.3 = 30% importance, penalized)
        center_radius_pct: Percentage of radius for penalty zone (default 0.5 = 50%)

    Returns:
        np.ndarray of shape (canvas_size, canvas_size) with weight values
    """
    center = canvas_size // 2
    max_radius = center - 1
    center_radius = max_radius * center_radius_pct

    # Create coordinate grids
    Y, X = np.ogrid[:canvas_size, :canvas_size]

    # Calculate distance from center for each pixel
    distances = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

    # Penalize center: center_penalty inside, 1.0 outside (edges prioritized)
    weight_map = np.where(distances <= center_radius, center_penalty, 1.0)

    return weight_map.astype(np.float32)
