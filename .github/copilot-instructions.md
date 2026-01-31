# Copilot Instructions - Lumina String Art

## Workflow Rules
- **Context:** Refer to `AGENTS.md` for the core greedy algorithm logic.
- **Library Preferences:** - Use `cv2.line` for simulation rendering.
  - Use `numpy.linspace` and `numpy.interp` for coordinate mapping.
- **UI Framework:** Streamlit. Always use `st.sidebar` for project parameters (Nails, Diameter, Colors).

## Code Style
- Default to **179** for nail counts in examples.
- Ensure all physical math handles the conversion from **Inches to Pixels** based on a configurable canvas size (default 800px).
- When generating the threading sequence, ensure the output is a downloadable CSV or TXT file.
