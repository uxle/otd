//! The P0000–P9999 phase registry (docs/01-ROADMAP-10000-PHASES.md).
//! P0130: the registry is data; the banner prints status at startup.

pub const TOTAL_PHASES: usize = 10_000;

pub struct PhaseEntry {
    pub id: &'static str,
    pub name: &'static str,
}

/// Phases implemented in this build (P0000–P0649, stage S0).
pub const IMPLEMENTED: &[PhaseEntry] = &[
    PhaseEntry { id: "P0000", name: "zero-dependency skeleton" },
    PhaseEntry { id: "P0010", name: "units + dimension checking" },
    PhaseEntry { id: "P0020", name: "Vec3/Mat4/AABB core" },
    PhaseEntry { id: "P0100", name: "lexer" },
    PhaseEntry { id: "P0110", name: "AST" },
    PhaseEntry { id: "P0120", name: "parser" },
    PhaseEntry { id: "P0130", name: "keyword registry (75 in 2.1)" },
    PhaseEntry { id: "P0140", name: "friendly errors" },
    PhaseEntry { id: "P0200", name: "mesh container" },
    PhaseEntry { id: "P0210", name: "cube/sphere" },
    PhaseEntry { id: "P0220", name: "cylinder/cone" },
    PhaseEntry { id: "P0230", name: "torus/pyramid/prism" },
    PhaseEntry { id: "P0240", name: "capsule/wedge/plane" },
    PhaseEntry { id: "P0300", name: "BSP union" },
    PhaseEntry { id: "P0310", name: "BSP subtract/intersect" },
    PhaseEntry { id: "P0320", name: "analytic hollow" },
    PhaseEntry { id: "P0400", name: "extrude" },
    PhaseEntry { id: "P0410", name: "revolve" },
    PhaseEntry { id: "P0420", name: "tube/helix" },
    PhaseEntry { id: "P0430", name: "3D text (dot font)" },
    PhaseEntry { id: "P0440", name: "terrain (fBm)" },
    PhaseEntry { id: "P0450", name: "metaballs" },
    PhaseEntry { id: "P0460", name: "sweep/loft" },
    PhaseEntry { id: "P0470", name: "STL/OBJ import" },
    PhaseEntry { id: "P0500", name: "volume/centroid/area" },
    PhaseEntry { id: "P0510", name: "slice area" },
    PhaseEntry { id: "P0520", name: "27 materials" },
    PhaseEntry { id: "P0530", name: "147 colors" },
    PhaseEntry { id: "P0540", name: "physics: mass/float/drop/collapse" },
    PhaseEntry { id: "P0550", name: "ask engine" },
    PhaseEntry { id: "P0560", name: "evaluator: patterns/templates" },
    PhaseEntry { id: "P0600", name: "SIMD software rasterizer" },
    PhaseEntry { id: "P0610", name: "asm! kernels" },
    PhaseEntry { id: "P0620", name: "own DEFLATE/PNG" },
    PhaseEntry { id: "P0630", name: "own HTTP/JSON server" },
    PhaseEntry { id: "P0640", name: "exporters + viewer" },
    // ---- OTD 2.0 "DEEP" (P0650–P1299, stage S1) ----
    PhaseEntry { id: "P0650", name: "OTD-ASM: 32-byte vector ISA" },
    PhaseEntry { id: "P0660", name: "expression bytecode compiler" },
    PhaseEntry { id: "P0670", name: "VM interpreter + asm! kernels" },
    PhaseEntry { id: "P0680", name: "--vm-dump disassembler" },
    PhaseEntry { id: "P0700", name: "VM evaluator integration + pi" },
    PhaseEntry { id: "P0800", name: "half-edge topology" },
    PhaseEntry { id: "P0810", name: "mesh welding" },
    PhaseEntry { id: "P0900", name: "DEC: cotan Laplacian + Taubin smooth" },
    PhaseEntry { id: "P1000", name: "Loop subdivision" },
    PhaseEntry { id: "P1050", name: "BVH acceleration structure" },
    PhaseEntry { id: "P1060", name: "SDF fields + smooth-min blend" },
    PhaseEntry { id: "P1070", name: "Freudenthal-Kuhn marching tets" },
    PhaseEntry { id: "P1080", name: "metaball manifold fix" },
    PhaseEntry { id: "P1150", name: "XPBD constraint solver" },
    PhaseEntry { id: "P1160", name: "catenary length solver" },
    PhaseEntry { id: "P1190", name: "rope() word + measured sag" },
    PhaseEntry { id: "P1180", name: "exact inertia tensor" },
    PhaseEntry { id: "P1250", name: "GGX + Smith + Schlick shading" },
    PhaseEntry { id: "P1260", name: "ray-traced soft shadows" },
    PhaseEntry { id: "P1270", name: "baked ambient occlusion" },
    PhaseEntry { id: "P1290", name: "deep-tier ask queries" },
    PhaseEntry { id: "P1295", name: "2.0 docs/examples/tests" },
    PhaseEntry { id: "P1300", name: "2.1 lexer: operators, ranges, sci-notation, block comments" },
    PhaseEntry { id: "P1310", name: "2.1 parser: precedence cascade, blocks, compound assign" },
    PhaseEntry { id: "P1320", name: "2.1 evaluator: loops, logic, templates, interpolation" },
    PhaseEntry { id: "P1330", name: "2.1 syntax audit: 8 bugs found and fixed" },
    PhaseEntry { id: "P1340", name: "2.1 docs/examples/lessons (staircase, orbit-tower)" },
];

pub fn banner() -> String {
    let done = IMPLEMENTED.len();
    format!(
        "OTD v{} — {} / {} phases ready",
        env!("CARGO_PKG_VERSION"), done, TOTAL_PHASES
    )
}
