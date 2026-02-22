"""
ShadowWeaver - Lumina String Art Generator

A Streamlit web application that converts images into physical string art
instructions for circular looms using a greedy pathfinding algorithm.
"""

import cv2
import numpy as np
import streamlit as st

from src.algorithm import GreedyStringArtAlgorithm
from src.config import Config
from src.export import generate_sequence_with_metadata, image_to_bytes
from src.geometry import calculate_nail_positions
from src.image_processing import get_display_image, preprocess_image
from src.renderer import StringArtRenderer

# Page configuration
st.set_page_config(
    page_title="ShadowWeaver - String Art Generator",
    page_icon="🎨",
    layout="wide",
)

st.title("ShadowWeaver String Art Generator")
st.markdown("*Convert images into physical string art instructions*")

# ============ SIDEBAR SETTINGS ============
st.sidebar.header("Configuration")

# Physical settings
num_nails = st.sidebar.number_input(
    "Number of Nails",
    min_value=50,
    max_value=500,
    value=Config.DEFAULT_NAIL_COUNT,
    step=1,
    help="Number of nails around the circular board",
)

diameter_inches = st.sidebar.number_input(
    "Board Diameter (inches)",
    min_value=5.0,
    max_value=36.0,
    value=Config.DEFAULT_DIAMETER_INCHES,
    step=0.25,
    help="Physical diameter of your circular board",
)

thread_thickness_mm = st.sidebar.number_input(
    "Thread Thickness (mm)",
    min_value=0.1,
    max_value=2.0,
    value=Config.DEFAULT_THREAD_THICKNESS_MM,
    step=0.05,
    help="Physical thickness of your thread",
)

min_nail_skip = st.sidebar.number_input(
    "Minimum Nail Skip",
    min_value=5,
    max_value=100,
    value=Config.DEFAULT_MIN_NAIL_SKIP,
    step=1,
    help="Minimum nail distance to prevent too-short lines",
)

# Color settings
st.sidebar.markdown("---")
st.sidebar.subheader("Colors")

background_color = st.sidebar.color_picker(
    "Background Color",
    Config.DEFAULT_BACKGROUND_COLOR,
    help="Color of the board background",
)

thread_color = st.sidebar.color_picker(
    "Thread Color 1",
    Config.DEFAULT_THREAD_COLOR,
    help="Primary thread color",
)

use_two_colors = st.sidebar.checkbox(
    "Use Two Thread Colors",
    value=False,
    help="Enable alternating between two thread colors for visual effect",
)

thread_color_2 = None

if use_two_colors:
    thread_color_2 = st.sidebar.color_picker(
        "Thread Color 2",
        "#FF6B6B",
        help="Secondary thread color (alternates with primary)",
    )

    # Lines for each color
    col1, col2 = st.sidebar.columns(2)
    with col1:
        lines_color_1 = st.number_input(
            "Lines (Color 1)",
            min_value=100,
            max_value=4000,
            value=Config.DEFAULT_TOTAL_LINES // 2,
            step=100,
            help="Number of lines for color 1",
        )
    with col2:
        lines_color_2 = st.number_input(
            "Lines (Color 2)",
            min_value=100,
            max_value=4000,
            value=Config.DEFAULT_TOTAL_LINES // 2,
            step=100,
            help="Number of lines for color 2",
        )
    st.sidebar.markdown(f"**Total Lines:** {lines_color_1 + lines_color_2}")
else:
    lines_color_1 = st.sidebar.number_input(
        "Total Lines",
        min_value=100,
        max_value=4000,
        value=Config.DEFAULT_TOTAL_LINES,
        step=100,
        help="Total number of thread lines to generate",
    )
    lines_color_2 = 0

invert_mode = st.sidebar.checkbox(
    "Invert Algorithm",
    value=False,
    help="Check for dark thread on light background (seeks dark pixels instead of bright)",
)

# Calculate derived values
canvas_size = Config.CANVAS_SIZE_PX
thread_thickness_px = Config.mm_to_pixels(
    thread_thickness_mm, diameter_inches, canvas_size
)
# Calculate realistic line opacity based on thread thickness (fixed, not adjustable)
line_opacity = Config.calculate_line_opacity(
    thread_thickness_mm, diameter_inches, canvas_size
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Thread Thickness:** {thread_thickness_px:.2f} pixels")
st.sidebar.markdown(f"**Line Opacity:** {line_opacity:.3f} (calculated from thickness)")
st.sidebar.markdown(f"**Canvas Size:** {canvas_size}px")

# ============ MAIN CONTENT ============

# Image uploader
uploaded_file = st.file_uploader(
    "Upload Target Image",
    type=["png", "jpg", "jpeg", "webp"],
    help="Upload the image you want to convert to string art",
)

if uploaded_file is not None:
    # Process image
    # invert=False (default): white thread on black background -> seek LIGHT areas (face/subject)
    # invert=True (checked): dark thread on white background -> seek DARK areas (invert to make them bright)
    image_bytes = uploaded_file.read()
    target_image = preprocess_image(image_bytes, canvas_size, invert=invert_mode)
    nail_positions = calculate_nail_positions(num_nails, canvas_size)

    # Display columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Target Image")
        # Show original image (undo inversion if it was applied)
        display_target = (255 - target_image) if invert_mode else target_image
        st.image(display_target, caption="Preprocessed Target", use_container_width=True)

    with col2:
        st.subheader("Thread Simulation")
        simulation_placeholder = st.empty()
        simulation_placeholder.info("Click 'Generate String Art' to start")

    # Generate button
    if st.button("Generate String Art", type="primary", use_container_width=True):
        # Initialize algorithm and renderer
        algorithm = GreedyStringArtAlgorithm(
            target_image=target_image,
            nail_positions=nail_positions,
            min_nail_skip=min_nail_skip,
            line_opacity=line_opacity,
            decay_factor=Config.DEFAULT_DECAY_FACTOR,
        )

        renderer = StringArtRenderer(
            canvas_size, background_color, thread_color, thread_color_2
        )
        renderer.create_blank_canvas()

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Store sequence
        sequence = [0]  # Start from nail 0

        # Track lines per color
        lines_drawn_color_1 = 0
        lines_drawn_color_2 = 0
        max_lines = lines_color_1 + lines_color_2

        # Run algorithm with progress updates
        for iteration, from_nail, to_nail in algorithm.run(max_lines, start_nail=0):
            # Determine which color to use
            if use_two_colors:
                # Alternate: even = color 1, odd = color 2
                color_index = iteration % 2

                # Skip if we've reached the limit for this color
                if color_index == 0 and lines_drawn_color_1 >= lines_color_1:
                    sequence.append(to_nail)
                    continue
                if color_index == 1 and lines_drawn_color_2 >= lines_color_2:
                    sequence.append(to_nail)
                    continue
            else:
                color_index = 0

            # Draw line
            p1 = tuple(nail_positions[from_nail])
            p2 = tuple(nail_positions[to_nail])
            renderer.draw_line(p1, p2, thread_thickness_px, line_opacity, color_index)

            # Track line counts
            if color_index == 0:
                lines_drawn_color_1 += 1
            else:
                lines_drawn_color_2 += 1

            sequence.append(to_nail)

            # Update progress (every 10 iterations for performance)
            total_drawn = lines_drawn_color_1 + lines_drawn_color_2
            if iteration % 10 == 0:
                progress = total_drawn / max_lines
                progress_bar.progress(progress)
                status_text.text(f"Processing line {total_drawn}/{max_lines}")

            # Update visualization periodically
            if iteration % 50 == 0:
                simulation_placeholder.image(
                    renderer.get_display_image(),
                    caption=f"Progress: {total_drawn} lines",
                    use_container_width=True,
                )

        # Final update
        progress_bar.progress(1.0)
        actual_lines = len(sequence) - 1
        status_text.text(f"Complete! {actual_lines} lines generated")

        final_image = renderer.get_display_image()
        simulation_placeholder.image(
            final_image, caption="Final Result", use_container_width=True
        )

        # Download buttons
        st.markdown("---")
        st.subheader("Download Results")

        download_col1, download_col2 = st.columns(2)

        with download_col1:
            sequence_data = generate_sequence_with_metadata(
                sequence, num_nails, diameter_inches, actual_lines
            )
            st.download_button(
                label="Download Nail Sequence (TXT)",
                data=sequence_data,
                file_name="string_sequence.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with download_col2:
            image_data = image_to_bytes(final_image)
            st.download_button(
                label="Download Simulation (PNG)",
                data=image_data,
                file_name="simulated_art.png",
                mime="image/png",
                use_container_width=True,
            )

else:
    st.info("Please upload an image to begin generating string art.")

    # Show sample nail layout
    st.subheader("Preview: Nail Layout")
    preview_nails = calculate_nail_positions(num_nails, canvas_size)

    # Parse colors for preview
    from src.renderer import hex_to_rgb

    bg_rgb = hex_to_rgb(background_color)
    thread_rgb = hex_to_rgb(thread_color)
    thread_rgb_2 = hex_to_rgb(thread_color_2) if thread_color_2 else thread_rgb

    preview = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    preview[:, :] = bg_rgb  # Fill with background color

    # Draw nails with thread color (alternating if two colors enabled)
    for i, (x, y) in enumerate(preview_nails):
        nail_color = thread_rgb_2 if (use_two_colors and i % 2 == 1) else thread_rgb
        cv2.circle(preview, (int(x), int(y)), 3, nail_color, -1)
        # Label every 20th nail for clarity
        if i % 20 == 0:
            # Use a contrasting gray for labels
            cv2.putText(
                preview,
                str(i),
                (int(x) + 5, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (128, 128, 128),
                1,
            )

    st.image(
        preview,
        caption=f"Nail Layout ({num_nails} nails)",
        use_container_width=True,
    )

# Footer
st.markdown("---")
st.markdown(
    "*ShadowWeaver uses a greedy algorithm to find optimal thread paths. "
    "Customize colors and invert mode in the sidebar.*"
)
