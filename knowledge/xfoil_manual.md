# XFOIL Command Reference Manual

XFOIL is an interactive program for the design and analysis of subsonic isolated airfoils. It consists of a collection of menu-driven routines which perform various useful functions such as:
- Viscous (or inviscid) analysis of an existing airfoil
- Airfoil design and redesign by interactive modification of surface speed distributions
- Airfoil redesign by interactive modification of geometric parameters
- Blending of airfoils
- Writing and reading of airfoil coordinates and polar save files
- Plotting of geometry, pressure distributions, and polars

## Top-Level Commands
These commands are available at the initial `XFOIL c>` prompt.

*   `LOAD [filename]`: Load an airfoil from a coordinate file (plain text, x y columns).
*   `NACA [digits]`: Generate a NACA 4 or 5 digit airfoil (e.g., `NACA 2412`).
*   `OPER`: Enter the **Operation** (Analysis) submenu.
*   `GDES`: Enter the **Geometry Design** submenu.
*   `MDES`: Enter the **QDES** (Inverse Design) submenu.
*   `PANE`: Re-panel the airfoil with a smoother distribution of nodes (crucial before analysis).
*   `PPAR`: Show/modify current paneling.
*   `SAVE [filename]`: Save the current airfoil geometry to a file.
*   `QUIT`: Exit XFOIL.

## OPER (Analysis) Submenu
Entered by typing `OPER` at the top level. Prompt becomes `.OPERi c>` (inviscid) or `.OPERv c>` (viscous).

### Setup
*   `VISC [Re]`: Toggle Viscous mode. Optionally specify Reynolds number immediately.
*   `Re [number]`: Set Reynolds number (e.g., `Re 1000000`).
*   `MACH [number]`: Set Mach number.
*   `ITER [n]`: Set maximum number of Newton iterations (default 10, suggest 50-100 for difficult convergence).
*   `INIT`: Reset boundary layer initialization (useful if solution diverges).

### Operating Conditions
*   `ALFA [deg]`: Calculate for a fixed Angle of Attack (e.g., `ALFA 5`).
*   `Cl [val]`: Calculate for a fixed Lift Coefficient (e.g., `Cl 0.5`).
*   `ASEQ [min] [max] [step]`: Run a sequence of Alphas (alpha sweep). E.g., `ASEQ -5 15 1`.
*   `CSEQ [min] [max] [step]`: Run a sequence of Cls.

### Data Management
*   `PACC`: Toggle "Polar Accumulation". Saves results to a file.
    1.  First prompt: Output filename (e.g., `polar_n2412.txt`).
    2.  Second prompt: Dump filename (e.g., press Enter to skip).
*   `PWRT`: Write current polar to a file (if not using PACC).

### Plotting
*   `CPx`: Plot Pressure Coefficient ($C_p$) vs x/c.
*   `BL`: Plot Boundary Layer characteristics.

## Common Troubleshooting
1.  **"NOT CONVERGED"**:
    - Cause: Flow separation or poor initialization.
    - Fix 1: Increase iterations (`ITER 100`).
    - Fix 2: Start from a simpler condition (alpha=0) and reach the target alpha in small steps using `ASEQ`.
    - Fix 3: Use `INIT` to reset the boundary layer state.

2.  **Jagged Cp distribution**:
    - Cause: Too few panels or poor coordinate resolution.
    - Fix: Go to top level, run `PANE`, then return to `OPER`.
