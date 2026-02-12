# Fundamentals of Aerodynamics (Based on Anderson)

## 1. Fundamental Variables
Aerodynamics is governed by four primary variables at any point in the flow field:
1.  **Pressure ($p$)**: Force per unit area acting normal to the surface.
2.  **Density ($\rho$)**: Mass per unit volume.
3.  **Temperature ($T$)**: Measure of average kinetic energy of particles.
4.  **Flow Velocity ($V$)**: Speed and direction of the fluid element.

## 2. Aerodynamic Forces
Forces on an airfoil stem from only two sources:
1.  **Pressure Distribution**: Acts normal to the surface.
2.  **Shear Stress ($\tau$)**: Acts tangential to the surface (due to friction).

- **Lift ($L$)**: The component of the net aerodynamic force perpendicular to the freestream velocity ($V_\infty$).
- **Drag ($D$)**: The component parallel to the freestream velocity.
- **Moment ($M$)**: Tendency to rotate, usually taken about the quarter-chord point ($c/4$).

## 3. Dimensionless Coefficients
To compare different airfoils and scales, we use coefficients:
$$ C_L = \frac{L}{q_\infty S} $$
$$ C_D = \frac{D}{q_\infty S} $$
$$ C_M = \frac{M}{q_\infty S c} $$
Where $q_\infty = \frac{1}{2}\rho V_\infty^2$ is the **dynamic pressure**.

## 4. Flow Regimes & Reynolds Number
The **Reynolds Number ($Re$)** is the ratio of inertial forces to viscous forces:
$$ Re = \frac{\rho V c}{\mu} $$

- **Laminar Flow**: Smooth, layered flow. Low skin friction, but easily separates.
- **Turbulent Flow**: Chaotic, mixed flow. High skin friction, but adheres better to the surface (resists separation) due to higher energy near the wall.
- **Transition**: The point where laminar boundary layer turns turbulent.

## 5. Circulation & Kutta-Joukowski Theorem
For a lifting cylinder or airfoil:
$$ L' = \rho_\infty V_\infty \Gamma $$
Where $\Gamma$ is the **circulation**. Lift is directly proportional to circulation. This is the theoretical basis for why curved airfoils generate lift (they induce net circulation).

## 6. Thin Airfoil Theory
- For symmetric airfoils: $c_l = 2\pi\alpha$ (Lift slope is $2\pi$ per radian).
- Aerodynamic Center: For thin airfoils, it is theoretically at the quarter-chord ($c/4$).

## 7. Compressibility
As speed increases ($Mach > 0.3$), density changes cannot be ignored.
- **Critical Mach Number ($M_{cr}$)**: The freestream Mach number at which flow over the airfoil first reaches sonic speed ($M=1$).
- **Drag Divergence**: Drag rises sharply shortly after $M_{cr}$ due to shock wave formation.
