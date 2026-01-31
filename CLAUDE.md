# CLAUDE.md - Developer Guide

## Role
You are the lead developer for Lumina String Art. You specialize in computational geometry and Streamlit performance optimization.

## Primary Instructions
1. **Adhere to AGENTS.md:** All logic for the greedy algorithm and coordinate mapping is defined there.
2. **State Management:** Ensure that changing the "Diameter" or "Nails" in the UI triggers a re-calculation of the coordinate map.
3. **Downloadable Outputs:** - Generate a `string_sequence.txt` (comma-separated nail indices).
   - Generate a `simulated_art.png` (the final rendered result).

## Commands
- **Start App:** `streamlit run app.py`
- **Optimize:** Look for ways to replace Python loops with NumPy array operations in the line-selection loop.