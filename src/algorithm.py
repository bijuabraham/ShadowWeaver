"""Greedy pathfinding algorithm for string art generation."""

from typing import Generator

import numpy as np

from .geometry import calculate_circular_distance, get_line_pixels


class GreedyStringArtAlgorithm:
    """
    Greedy pathfinding algorithm for string art generation.

    Strategy:
    - Black background, white thread
    - Select paths that cover highest-intensity pixels in target
    - Apply simulated decay after each line to prevent re-selection
    """

    def __init__(
        self,
        target_image: np.ndarray,
        nail_positions: np.ndarray,
        min_nail_skip: int = 20,
        line_opacity: float = 0.1,
        decay_factor: float = 0.85,
    ):
        """
        Initialize the algorithm.

        Args:
            target_image: Grayscale target image (0-255, higher = more thread needed)
            nail_positions: Array of (x, y) coordinates for each nail
            min_nail_skip: Minimum circular distance between consecutive nails
            line_opacity: How much white each line adds (for reference)
            decay_factor: How much to reduce target intensity after line (0.0-1.0)
        """
        self.target = target_image.astype(np.float32) / 255.0
        self.nails = nail_positions
        self.num_nails = len(nail_positions)
        self.min_skip = min_nail_skip
        self.line_opacity = line_opacity
        self.decay_factor = decay_factor
        self.sequence: list[int] = []

        # Pre-compute valid nail pairs (respecting min_skip)
        self._precompute_valid_connections()

    def _precompute_valid_connections(self) -> None:
        """Pre-compute valid target nails for each source nail."""
        self.valid_targets: dict[int, np.ndarray] = {}
        for i in range(self.num_nails):
            valid = []
            for j in range(self.num_nails):
                dist = calculate_circular_distance(i, j, self.num_nails)
                if dist >= self.min_skip:
                    valid.append(j)
            self.valid_targets[i] = np.array(valid, dtype=np.int32)

    def evaluate_line(self, nail_a: int, nail_b: int) -> float:
        """
        Evaluate the score of a potential line.

        Score = weighted average intensity of pixels along the line.
        Center pixels are weighted higher to prioritize facial features.
        Higher score = more desirable line (more thread needed in that area).

        Args:
            nail_a: Starting nail index
            nail_b: Ending nail index

        Returns:
            Average intensity along the line
        """
        p1 = tuple(self.nails[nail_a])
        p2 = tuple(self.nails[nail_b])

        # Get pixels along the line
        pixels = get_line_pixels(p1, p2)

        # Sample target image at these pixels (bounds checking)
        x_coords = np.clip(pixels[:, 0], 0, self.target.shape[1] - 1)
        y_coords = np.clip(pixels[:, 1], 0, self.target.shape[0] - 1)

        # Get intensities along line
        intensities = self.target[y_coords, x_coords]

        # Average intensity along the line
        return float(np.mean(intensities))

    def apply_decay(self, nail_a: int, nail_b: int) -> None:
        """
        Reduce intensity in target image along the drawn line.

        Prevents algorithm from selecting the same path repeatedly.

        Args:
            nail_a: Starting nail index
            nail_b: Ending nail index
        """
        p1 = tuple(self.nails[nail_a])
        p2 = tuple(self.nails[nail_b])

        pixels = get_line_pixels(p1, p2)
        x_coords = np.clip(pixels[:, 0], 0, self.target.shape[1] - 1)
        y_coords = np.clip(pixels[:, 1], 0, self.target.shape[0] - 1)

        # Reduce intensity (simulated coverage)
        self.target[y_coords, x_coords] *= self.decay_factor

    def find_best_next_nail(self, current_nail: int) -> tuple[int, float]:
        """
        Find the best next nail from current position.

        Args:
            current_nail: Current nail index

        Returns:
            Tuple of (best_nail_index, score)
        """
        valid_nails = self.valid_targets[current_nail]

        if len(valid_nails) == 0:
            return -1, 0.0

        best_nail = -1
        best_score = -1.0

        for nail in valid_nails:
            score = self.evaluate_line(current_nail, nail)
            if score > best_score:
                best_score = score
                best_nail = int(nail)

        return best_nail, best_score

    def run(
        self,
        total_lines: int,
        start_nail: int = 0,
    ) -> Generator[tuple[int, int, int], None, None]:
        """
        Run the greedy algorithm.

        Yields:
            Tuple of (iteration, from_nail, to_nail) for progress tracking
        """
        current_nail = start_nail
        self.sequence = [current_nail]

        for i in range(total_lines):
            next_nail, score = self.find_best_next_nail(current_nail)

            # Stop if no valid paths or negligible score
            if next_nail == -1 or score <= 0.001:
                break

            # Apply decay before yielding (target updated for next iteration)
            self.apply_decay(current_nail, next_nail)

            yield (i, current_nail, next_nail)

            self.sequence.append(next_nail)
            current_nail = next_nail

    def get_sequence(self) -> list[int]:
        """
        Return the complete nail sequence.

        Returns:
            List of nail indices in order of traversal
        """
        return self.sequence


class GradientDescentStringArtAlgorithm:
    """
    Gradient Descent algorithm for string art generation.

    Uses continuous weight optimization:
    - Treats each line as a weight to be optimized
    - Minimizes MSE between target and reconstructed image
    - Uses projected gradient descent with non-negativity constraint
    """

    def __init__(
        self,
        target_image: np.ndarray,
        nail_positions: np.ndarray,
        min_nail_skip: int = 20,
        line_opacity: float = 0.1,
        learning_rate: float = 0.01,
    ):
        """
        Initialize the algorithm.

        Args:
            target_image: Grayscale target image (0-255, higher = more thread needed)
            nail_positions: Array of (x, y) coordinates for each nail
            min_nail_skip: Minimum circular distance between consecutive nails
            line_opacity: How much white each line adds
            learning_rate: Gradient descent learning rate
        """
        self.target = target_image.astype(np.float32) / 255.0
        self.nails = nail_positions
        self.num_nails = len(nail_positions)
        self.min_skip = min_nail_skip
        self.line_opacity = line_opacity
        self.learning_rate = learning_rate
        self.sequence: list[int] = []

        # Pre-compute all valid lines and their masks
        self._precompute_line_masks()

    def _precompute_line_masks(self) -> None:
        """Pre-compute all valid line masks as coordinate lists."""
        self.line_masks: list[tuple[np.ndarray, np.ndarray]] = []
        self.line_indices: list[tuple[int, int]] = []

        for i in range(self.num_nails):
            for j in range(self.num_nails):
                if i == j:
                    continue
                dist = calculate_circular_distance(i, j, self.num_nails)
                if dist >= self.min_skip:
                    p1 = tuple(self.nails[i])
                    p2 = tuple(self.nails[j])
                    pixels = get_line_pixels(p1, p2)
                    x_coords = np.clip(pixels[:, 0], 0, self.target.shape[1] - 1)
                    y_coords = np.clip(pixels[:, 1], 0, self.target.shape[0] - 1)
                    self.line_masks.append((y_coords, x_coords))
                    self.line_indices.append((i, j))

        self.num_lines = len(self.line_masks)
        # Initialize weights to 0
        self.weights = np.zeros(self.num_lines, dtype=np.float32)

    def _compute_image(self) -> np.ndarray:
        """Compute the current image from line weights."""
        canvas = np.zeros_like(self.target)
        for idx, (y_coords, x_coords) in enumerate(self.line_masks):
            canvas[y_coords, x_coords] += self.line_opacity * self.weights[idx]
        return np.clip(canvas, 0, 1)

    def _compute_gradient(self, error: np.ndarray) -> np.ndarray:
        """Compute gradient for all weights."""
        gradients = np.zeros(self.num_lines, dtype=np.float32)
        for idx, (y_coords, x_coords) in enumerate(self.line_masks):
            # Gradient = sum of error values along the line
            gradients[idx] = np.sum(error[y_coords, x_coords]) * self.line_opacity
        return gradients

    def run(
        self,
        total_lines: int,
        start_nail: int = 0,
    ) -> Generator[tuple[int, int, int], None, None]:
        """
        Run gradient descent optimization.

        Yields:
            Tuple of (iteration, from_nail, to_nail) for progress tracking
        """
        # Calculate target as difference from background (1.0 = white thread, 0.0 = background)
        target = self.target

        # Run gradient descent iterations
        max_iterations = total_lines * 10  # More iterations than output lines

        for iteration in range(max_iterations):
            # Forward pass
            current_image = self._compute_image()

            # Compute error
            error = target - current_image

            # Compute gradient
            gradients = self._compute_gradient(error)

            # Update weights with projected gradient descent
            self.weights -= self.learning_rate * gradients

            # Project to non-negative (can't draw negative lines)
            self.weights = np.maximum(self.weights, 0)

            # Yield progress periodically
            if iteration % (total_lines) == 0:
                mse = np.mean(error ** 2)
                yield (-1, -1, -1)  # Special marker for progress

        # Convert weights to sequence: take top weighted lines
        self._create_sequence_from_weights(total_lines, start_nail)

        # Yield the sequence
        for i in range(len(self.sequence) - 1):
            yield (i, self.sequence[i], self.sequence[i + 1])

    def _create_sequence_from_weights(self, total_lines: int, start_nail: int) -> None:
        """Convert weights to a valid nail sequence."""
        # Get indices of top-weighted lines
        top_indices = np.argsort(self.weights)[::-1][:total_lines]

        # Build sequence: start from start_nail, greedily connect
        self.sequence = [start_nail]
        current = start_nail

        for _ in range(len(top_indices)):
            best_next = -1
            best_weight = -1

            for idx in top_indices:
                line_from, line_to = self.line_indices[idx]
                if line_from == current and self.weights[idx] > best_weight:
                    best_next = line_to
                    best_weight = self.weights[idx]
                elif line_to == current and self.weights[idx] > best_weight:
                    best_next = line_from
                    best_weight = self.weights[idx]

            if best_next == -1:
                # Try to find any unused line that connects
                for idx in top_indices:
                    line_from, line_to = self.line_indices[idx]
                    if line_from == current or line_to == current:
                        best_next = line_to if line_from == current else line_from
                        break

            if best_next == -1:
                break

            self.sequence.append(best_next)
            current = best_next

    def get_sequence(self) -> list[int]:
        """Return the complete nail sequence."""
        return self.sequence
