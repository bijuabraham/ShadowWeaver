"""Export utilities for string art outputs."""

import cv2
import numpy as np


def generate_sequence_file(sequence: list[int]) -> bytes:
    """
    Generate downloadable nail sequence as comma-separated text.

    Args:
        sequence: List of nail indices

    Returns:
        UTF-8 encoded bytes of comma-separated values
    """
    content = ",".join(map(str, sequence))
    return content.encode("utf-8")


def generate_sequence_with_metadata(
    sequence: list[int],
    num_nails: int,
    diameter: float,
    total_lines: int,
) -> bytes:
    """
    Generate detailed sequence file with metadata header.

    Args:
        sequence: List of nail indices
        num_nails: Number of nails on the board
        diameter: Physical board diameter in inches
        total_lines: Number of lines generated

    Returns:
        UTF-8 encoded bytes with header and sequence
    """
    lines = [
        "# ShadowWeaver String Art Sequence",
        f"# Nails: {num_nails}",
        f"# Diameter: {diameter} inches",
        f"# Total Lines: {total_lines}",
        f"# Sequence Length: {len(sequence)}",
        "#",
        "# Nail sequence (comma-separated):",
        ",".join(map(str, sequence)),
    ]
    return "\n".join(lines).encode("utf-8")


def image_to_bytes(image: np.ndarray, format: str = "PNG") -> bytes:
    """
    Convert numpy image array to bytes for download.

    Args:
        image: Grayscale or RGB image array
        format: Image format (PNG, JPG, etc.)

    Returns:
        Encoded image bytes

    Raises:
        ValueError: If encoding fails
    """
    success, encoded = cv2.imencode(f".{format.lower()}", image)
    if success:
        return encoded.tobytes()
    raise ValueError("Failed to encode image")
