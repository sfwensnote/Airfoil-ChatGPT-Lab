# Engineering Fluid Mechanics (Based on White)

## 1. Fluid Properties
- **Viscosity ($\mu$)**: A fluid's resistance to shear/deformation.
    - **Newtonian Fluids**: Shear stress is linearly proportional to strain rate ($\tau = \mu \frac{du}{dy}$). E.g., Air, Water.
    - **Non-Newtonian**: Relationship is non-linear (e.g., blood, ketchup).
- **No-Slip Condition**: Fluid at a solid boundary has zero velocity relative to the boundary.

## 2. Conservation Laws (Integral Form)
Fluid mechanics relies on three conservation principles applied to a Control Volume (CV):
1.  **Conservation of Mass** (Continuity Equation): Mass in - Mass out = Change of mass inside.
2.  **Conservation of Momentum**: Net force on CV = Rate of change of momentum.
    $$ \sum F = \frac{d}{dt} \int_{CV} V \rho d\mathcal{V} + \int_{CS} V (\rho V \cdot dA) $$
3.  **Conservation of Energy**: Energy cannot be created or destroyed.

## 3. Navier-Stokes Equations
The differential form of momentum conservation for viscous fluids.
$$ \rho \frac{D\mathbf{V}}{Dt} = -\nabla p + \rho \mathbf{g} + \mu \nabla^2 \mathbf{V} $$
(Inertia forces) = (Pressure grad) + (Gravity) + (Viscous forces)

These equations are generally unsolvable analytically for complex geometries (like airfoils) and require **CFD (Computational Fluid Dynamics)** to solve numerically.

## 4. Boundary Layer Theory (Prandtl)
- Near a solid surface, viscous effects are dominant in a thin layer called the **Boundary Layer**.
- Outside this layer, the flow acts essentially inviscid.
- **Significance**: Drag and heat transfer occur primarily within this layer.
- **Separation**: If pressure rises too fast (adverse pressure gradient), the slow-moving boundary layer can stop and reverse direction, causing the flow to detach from the surface. This is the mechanism behind **Stall**.

## 5. Bernoulli's Equation
For steady, incompressible, inviscid, irrotational flow along a streamline:
$$ p + \frac{1}{2}\rho V^2 + \rho g h = \text{constant} $$
- **Static Pressure** ($p$): Actual thermodynamic pressure.
- **Dynamic Pressure** ($\frac{1}{2}\rho V^2$): Pressure due to motion.
- **Stagnation Pressure** ($p_0$): Pressure if fluid is brought to rest ($p + \frac{1}{2}\rho V^2$).

**Misconception Alert**: Bernoulli applies *along* a streamline or in irrotational flow. It explains lift via pressure differences caused by velocity differences, but does not explain *why* the velocity differences exist (that requires Circulation/Kutta condition).
