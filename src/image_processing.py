"""Image preprocessing utilities for string art generation."""

import cv2
import numpy as np
import streamlit as st


def create_circular_mask(size: int) -> np.ndarray:
    """
    Create a circular mask for the canvas.

    Args:
        size: Size of the square mask

    Returns:
        Binary mask with circle filled (uint8, 0 or 255)
    """
    Y, X = np.ogrid[:size, :size]
    center = size // 2
    radius = size // 2 - 1
    mask = ((X - center) ** 2 + (Y - center) ** 2 <= radius ** 2).astype(np.uint8) * 255
    return mask


@st.cache_data
def preprocess_image(
    image_bytes: bytes, canvas_size: int = 800, invert: bool = True
) -> np.ndarray:
    """
    Preprocess uploaded image for string art algorithm.

    Steps:
    1. Decode image from bytes
    2. Convert to grayscale
    3. Resize to fit within circular canvas
    4. Apply circular mask
    5. Optionally invert based on thread/background combination

    Args:
        image_bytes: Raw image bytes from upload
        canvas_size: Target canvas size in pixels
        invert: If True, invert image (for white thread on dark background).
                If False, keep original (for dark thread on light background).

    Returns:
        np.ndarray: Grayscale image (0-255) where higher = more thread desired
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Failed to decode image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize maintaining aspect ratio, then center crop to square
    h, w = gray.shape
    scale = canvas_size / min(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Center crop to square
    start_x = (new_w - canvas_size) // 2
    start_y = (new_h - canvas_size) // 2

    # Handle edge case where resize produces exact size
    start_x = max(0, start_x)
    start_y = max(0, start_y)

    cropped = resized[start_y : start_y + canvas_size, start_x : start_x + canvas_size]

    # Ensure exact size (pad if needed)
    if cropped.shape[0] != canvas_size or cropped.shape[1] != canvas_size:
        padded = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
        ph, pw = cropped.shape
        padded[:ph, :pw] = cropped
        cropped = padded

    # Apply circular mask
    mask = create_circular_mask(canvas_size)
    masked = cv2.bitwise_and(cropped, mask)

    # Invert based on mode:
    # - invert=False (default): white thread on black background -> seek LIGHT areas (faces) -> don't invert
    # - invert=True: dark thread on white background -> seek DARK areas -> invert to make them bright
    if invert:
        return 255 - masked
    else:
        return masked


def get_display_image(target_image: np.ndarray) -> np.ndarray:
    """
    Convert processed target image back to displayable format.

    Args:
        target_image: Inverted grayscale target

    Returns:
        Original-looking grayscale image for display
    """
    return 255 - target_image
