# AGENTS.md - Lumina String Art Core Logic

## Project Overview
A Streamlit web application that converts images into physical string art instructions for circular looms.

## Dynamic Configuration (User-Adjustable)
The app must allow users to configure the following in a Sidebar:
- **Nail Count:** (Default: 179)
- **Physical Diameter:** (Default: 13.75 inches)
- **Colors:** Support custom hex codes for both Thread and Background.
- **Algorithm Weights:** Line opacity (alpha) and total iterations.

## Technical Implementation Standards
1. **Coordinate Mapping:** Calculate nail positions using:
   $x = R \cos(\theta)$, $y = R \sin(\theta)$.
2. **The "Greedy" Pathfinding:** - Iterate through possible lines from the current nail.
   - Select the line that maximizes pixel intensity alignment.
   - For a Black background, seek high-brightness pixels.
   - For a White background, seek low-brightness (dark) pixels.
3. **Simulated Decay:** After placing a line, modify the target image by the `line_opacity` to prevent the algorithm from looping on the same path.

## Performance
- Use **NumPy vectorization** for line-integral calculations.
- Use **Streamlit Caching** (`@st.cache_data`) for image preprocessing and nail coordinate generation.