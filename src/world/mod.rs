pub mod eval;
pub mod shapes;
pub mod materials;
pub mod colors;
pub mod physics;
pub mod ask;
// P0650+ — the 2.0 Deep tier
pub mod xpbd;
pub mod rope;
// P1420 — animation: spin parts / turn the scene
pub mod anim;
// P1420b — settle: rigid gravity + real solidity + environments
pub mod settle;
// P1430 — chemistry: liquid miscibility, gas mixing, reactions
pub mod chem;
// ---- OTD3: the science expansion (P2100 series) ----
// P2100 — the complete energy taxonomy: 2 categories, 9 forms
pub mod energy;
// P2110 — thermodynamics: temperature, heat, phase, the four laws
pub mod thermo;
// P2120 — magnetism: fields, forces, Faraday, Ampère, Earth
pub mod magnetism;
// P2130 — waves: sound and light, Doppler, Snell, Wien, echo ranging
pub mod waves;
// P2140 — time & motion: kinematics, clocks, relativity
pub mod time;
// P2150 — screw animation: the nut-and-bolt machine
pub mod screw;
// ---- OTD3.1: the self-make expansion (P2200 series) ----
// P2220 — statistics & probability: the science of uncertainty
pub mod stats;
// P2230 — astronomy: Kepler, orbits, stars, and the relativistic floor
pub mod astro;
// ---- OTD3.3: the subatomic layer (P2250 series) ----
// P2250 — the Standard Model: quarks, gluons, photons, neutrinos
pub mod particles;
// P2260 — atoms & nuclei: the periodic table, shells, binding, decay
pub mod atom;
// ---- OTD6: the motor + circuit simulation ----
pub mod motor;
// ---- OTD4: the dynamics expansion (P2300 series) ----
// P2300 — aerodynamics: drag, lift, terminal velocity, Reynolds, Mach
pub mod aerodynamics;
// P2310 — fluid dynamics: continuity, Bernoulli, Poiseuille, Stokes
pub mod fluiddynamics;
// P2320 — electrodynamics: Ohm, Kirchhoff, RC/RL/LC, Maxwell's light
pub mod electrodynamics;
// P2330 — stellar dynamics: N-body, virial theorem, Jeans scale
pub mod stellardynamics;
// P2340 — rigid body dynamics: inertia tensor, angular momentum, gyroscopes
pub mod rigidbody;

pub use eval::{ConsoleLine, LineKind, Part, Stats, World};
