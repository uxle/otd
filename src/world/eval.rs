//! P0560 — the evaluator: AST → World (parts, stats, console, errors).

use crate::geo::csg::{mesh_intersect, mesh_subtract, mesh_union};
use crate::geo::hollow::{hollow, HollowResult, Open};
use crate::geo::mesh::{Kind, Mesh};
use crate::lang::ast::*;
use crate::lang::errors::Error;
use crate::lang::keywords as kw;
use crate::lang::parser;
use crate::math3::{Aabb, M4, V3};
use crate::units::{Dim, Qty};
use crate::world::ask;
use crate::world::colors::{self, Color};
use crate::world::materials::{self, Material};

#[derive(Clone, Debug)]
pub enum Val {
    Qty(Qty),
    Str(String),
    /// 2 or 3 quantities (positions in mm, angles in deg)
    Tuple(Vec<Qty>),
    /// list of values (profiles / paths / sections)
    List(Vec<Val>),
    /// true / false (2.1)
    Bool(bool),
    /// a..b inclusive — plain numbers only (2.1)
    Range(f64, f64),
    Shape(ShapeVal),
    /// enum words: x/y/z, top/bottom/none, on/off, earth/moon/mars…
    Word(String),
}

#[derive(Clone, Debug, Default)]
pub struct ShapeVal {
    pub meshes: Vec<Mesh>,
    pub kinds: Vec<Kind>,
    pub mat: Option<&'static Material>,
    pub color: Option<Color>,
}

impl ShapeVal {
    pub(super) fn from_mesh(mesh: Mesh, kind: Kind) -> ShapeVal {
        ShapeVal { meshes: vec![mesh], kinds: vec![kind], ..Default::default() }
    }
    pub fn bbox(&self) -> Aabb {
        let mut bb = Aabb::empty();
        for m in &self.meshes {
            if !m.is_empty() {
                bb.grow_box(&m.bbox());
            }
        }
        bb
    }
    fn is_empty(&self) -> bool {
        self.meshes.iter().all(|m| m.is_empty())
    }
    fn translate(&mut self, t: V3) {
        let m = M4::translate(t.x(), t.y(), t.z());
        for mesh in &mut self.meshes {
            mesh.transform(&m);
        }
    }
    fn merged_mesh(&self) -> Mesh {
        let mut m = Mesh::new();
        for x in &self.meshes {
            m.merge(x);
        }
        m
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LineKind {
    Answer,
    Print,
    Warn,
    Sim,
    Info,
    Error,
}

#[derive(Clone, Debug)]
pub struct ConsoleLine {
    pub kind: LineKind,
    pub text: String,
}

#[derive(Clone)]
pub struct Part {
    pub name: String,
    pub mesh: Mesh,
    pub material: Option<&'static Material>,
    pub color: Option<Color>,
    pub hidden: bool,
    // measured truth
    pub volume_mm3: f64,
    pub mass_g: f64,
    pub centroid: Option<V3>,
    pub area_mm2: f64,
    /// OTD4 — when true, this part is a permanent magnet (overrides its
    /// material's intrinsic magnetism for `simulate: magnet`). The moment
    /// (A·m²) is stored on the world's magnet registry below.
    pub magnetized: bool,
}

#[derive(Clone)]
pub struct Stats {
    pub total_volume_mm3: f64,
    pub total_mass_g: f64,
    pub total_area_mm2: f64,
    pub com: Option<V3>,
    pub bbox: Aabb,
    pub visible_parts: usize,
    pub overlaps: bool,
    pub avg_density_kg_m3: f64,
}

impl Default for Stats {
    fn default() -> Stats {
        Stats {
            total_volume_mm3: 0.0,
            total_mass_g: 0.0,
            total_area_mm2: 0.0,
            com: None,
            bbox: Aabb::empty(),
            visible_parts: 0,
            overlaps: false,
            avg_density_kg_m3: 0.0,
        }
    }
}

#[derive(Default)]
pub struct World {
    pub title: String,
    pub parts: Vec<Part>,
    pub console: Vec<ConsoleLine>,
    pub errors: Vec<Error>,
    pub gravity: f64,
    pub camera: Option<String>,
    pub version: Option<f64>,
    pub stats: Stats,
    pub exports: Vec<(String, String)>,
    pub default_material: Option<&'static Material>,
    pub default_color: Option<Color>,
    /// measured sag (mm) of the last rope() — powers ask "cable sag?" (2.0)
    pub rope_sag: Option<f64>,
    /// P1420b — the medium the scene lives in: "" (= air), vacuum, water,
    /// oil, or a density number. Buoyancy + drag + gas chemistry read it.
    pub environment: String,
    /// P2110 — the scene's temperature, °C (drives heat, sound speed, phase).
    /// Default 20 °C — a pleasant lab. Set with `temperature: 800` or
    /// `temperature: 350K` / `temperature: 72F`.
    pub temp_c: f64,
    /// OTD4 P2300 — strict mode: when true, solid-solid interpenetration
    /// FAILS compilation (instead of just emitting a warning). Toggle with
    /// `strict: overlap` / `strict: off`. The silent-wrongness fix.
    pub strict_overlap: bool,
    /// OTD4 P2310 — explicit magnetic moments (A·m²) for parts marked with
    /// `magnetize: name [moment: <expr>]`. Keyed by part name. When a part
    /// is in this map, its moment is used by `simulate: magnet` regardless
    /// of the material's intrinsic class.
    pub magnet_moments: std::collections::HashMap<String, f64>,
    /// OTD6 #5 — electrical connections: (partA, partB) pairs declared
    /// with `connect: A B`. Read by `simulate: circuit` to walk the real
    /// resistance of modeled windings and report current/voltage drop.
    pub connections: Vec<(String, String)>,
}

struct Entry {
    name: String,
    sv: ShapeVal,
    hidden: bool,
}

#[derive(Clone)]
struct Template {
    params: Vec<String>,
    /// `define name(params) = expr` form
    body: Option<Expr>,
    /// `define name(params) … end` multi-statement form (2.1)
    stmts: Vec<Stmt>,
    line: usize,
}

pub(super) struct Ctx<'a> {
    pub(super) env: std::collections::HashMap<String, Val>,
    templates: std::collections::HashMap<String, Template>,
    magic: Vec<(String, Qty)>,
    pub(super) errors: Vec<Error>,
    pub(super) world: &'a mut World,
    entries: Vec<Entry>,
    default_unit: String,
    /// scene default unit as a factor to millimeters (bare numbers × this)
    pub(super) unit_mm: f64,
    depth: u32, // recursion guard for templates
    // ---- 2.1: control flow bookkeeping ----
    /// nesting depth of for/while bodies (break/continue legality)
    in_loop: u32,
    /// inside a define … end template body: statements don't become parts
    tpl_mode: bool,
    /// the last value a define … end body produced
    tpl_val: Option<Val>,
    /// env write journal for template calls (first-write wins, restored after)
    journal: Option<Vec<(String, Option<Val>)>>,
    /// P0650: compiled-VM cache keyed by AST node address — pattern bodies
    /// compile once and re-run every iteration with fresh magic vars
    vm_cache: std::collections::HashMap<usize, crate::vm::compile::Program>,
    /// AST nodes that failed VM compilation (shape calls etc.) — don't retry
    vm_duds: std::collections::HashSet<usize>,
    /// measured sag of the most recent rope() (mm) — ask "cable sag?"
    pub(super) rope_sag: Option<f64>,
}

/// What a statement decided to do to the enclosing loop (2.1).
#[derive(Clone, Copy, PartialEq)]
enum Flow {
    Normal,
    Break,
    Continue,
}

impl<'a> Ctx<'a> {
    pub(super) fn err(&mut self, line: usize, msg: impl Into<String>) -> Error {
        let e = Error::new(line, msg);
        self.errors.push(e.clone());
        e
    }
    pub(super) fn err_hint(&mut self, line: usize, msg: impl Into<String>, hint: impl Into<String>) -> Error {
        let e = Error::new(line, msg).with_hint(hint);
        self.errors.push(e.clone());
        e
    }
}

/// Compile OTD source into a World.
pub fn compile(src: &str) -> World {
    // pre-scan for the default unit (affects bare-number lexing)
    let default_unit = scan_unit(src).unwrap_or_else(|| "cm".into());
    let prog = parser::parse(src, &default_unit);

    let mut world = World {
        title: "Untitled".into(),
        gravity: crate::units::G_EARTH,
        temp_c: 20.0,
        ..Default::default()
    };
    world.errors.extend(prog.errors.clone());

    let unit_mm = crate::units::unit_factor(&default_unit).map(|(f, _)| f).unwrap_or(10.0);
    let mut ctx = Ctx {
        env: std::collections::HashMap::new(),
        templates: std::collections::HashMap::new(),
        magic: Vec::new(),
        errors: prog.errors,
        world: &mut world,
        entries: Vec::new(),
        default_unit,
        unit_mm,
        depth: 0,
        in_loop: 0,
        tpl_mode: false,
        tpl_val: None,
        journal: None,
        vm_cache: std::collections::HashMap::new(),
        vm_duds: std::collections::HashSet::new(),
        rope_sag: None,
    };

    let mut deferred: Vec<Stmt> = Vec::new();
    for stmt in prog.stmts {
        match stmt {
            Stmt::Simulate(..) | Stmt::Ask(..) | Stmt::Particle(..) => deferred.push(stmt),
            other => {
                // P0650: AST nodes are freed statement-by-statement (exec
                // consumes them by value), so heap addresses get recycled —
                // the VM cache must not outlive the statement it serves.
                // Pattern bodies compile once WITHIN their statement, which
                // is exactly the hot loop the cache exists for.
                ctx.vm_cache.clear();
                ctx.vm_duds.clear();
                let _ = ctx.exec(other);
            }
        }
    }

    ctx.world.rope_sag = ctx.rope_sag;

    // finalize entries → parts (scene defaults fill un-overridden props)
    let def_mat = ctx.world.default_material;
    let def_col = ctx.world.default_color;
    let mut parts: Vec<Part> = Vec::new();
    let mut any_unmaterialized = false;
    for e in ctx.entries.drain(..) {
        let material = e.sv.mat.or(def_mat);
        let color = e.sv.color.or(def_col);
        if e.sv.is_empty() {
            continue;
        }
        if material.is_none() {
            any_unmaterialized = true;
        }
        let mesh = if e.sv.meshes.len() == 1 {
            e.sv.meshes.into_iter().next().unwrap()
        } else {
            let mut m = Mesh::new();
            for x in &e.sv.meshes {
                m.merge(x);
            }
            m
        };
        let volume = mesh.volume_signed().abs();
        let density = material.map(|m| m.density).unwrap_or(1050.0);
        let mass_g = volume * 1e-9 * density * 1000.0; // mm³ × 1e-9 = m³; × kg/m³ = kg; × 1000 = g
        parts.push(Part {
            name: e.name,
            volume_mm3: volume,
            mass_g,
            centroid: mesh.centroid(),
            area_mm2: mesh.area(),
            material,
            color: color.or(material.map(|m| Color::new(m.color[0], m.color[1], m.color[2]))),
            hidden: e.hidden,
            mesh,
            magnetized: false,
        });
    }
    ctx.world.parts = parts;
    ctx.world.errors = ctx.errors;

    finalize_stats(ctx.world);

    // OTD4 — material suggestion: if any part has no material, recommend
    // common ones based on what the scene seems to be making. The
    // silent-wrongness fix: a part with no material defaults to a plastic-
    // class density (1050 kg/m³) which is rarely what the user wanted.
    if any_unmaterialized {
        let n_unmat = ctx.world.parts.iter().filter(|p| p.material.is_none()).count();
        let first_unmat = ctx.world.parts.iter().find(|p| p.material.is_none()).map(|p| p.name.clone()).unwrap_or_default();
        ctx.world.console.push(ConsoleLine {
            kind: LineKind::Warn,
            text: format!(
                "{} part(s) (e.g., '{}') have no material — mass defaults to plastic density 1050 kg/m³. Add `material: <name>` to fix. Common choices:",
                n_unmat, first_unmat
            ),
        });
        ctx.world.console.push(ConsoleLine {
            kind: LineKind::Info,
            text: "  steel (structural) · aluminum (lightweight) · copper (conductor) · glass (transparent) · oak (wood) · ceramic (cup) · water (liquid) · iron (magnet)".into(),
        });
        ctx.world.console.push(ConsoleLine {
            kind: LineKind::Info,
            text: "  or set a scene-wide default:  material: steel  (applies to every part that doesn't override)".into(),
        });
    }
    // OTD4 — color suggestion: similar friendly hint for missing colors
    let any_uncolored = ctx.world.parts.iter().any(|p| p.color.is_none() && p.material.is_none());
    if any_uncolored {
        ctx.world.console.push(ConsoleLine {
            kind: LineKind::Info,
            text: "tip: parts without a color inherit from their material (steel → light gray, oak → tan). Add `color: <name>` (like color: ivory) or a hex `color: #1e90ff` to override.".into(),
        });
    }

    // deferred ask/simulate need the finished world
    let mut w = std::mem::take(ctx.world);
    for stmt in deferred {
        match stmt {
            Stmt::Ask(q, line) => {
                if let Some(a) = ask::answer(&q, &w, line) {
                    w.console.push(a);
                }
            }
            Stmt::Simulate(sim, line) => {
                let g = w.gravity;
                let mut lines = crate::world::physics::simulate_mut(&sim, &mut w, line, g);
                w.console.append(&mut lines);
                // settling / gas / mixing MOVE or MERGE geometry — re-measure
                if matches!(sim.as_str(), "settle" | "gas" | "mix") {
                    finalize_stats(&mut w);
                }
            }
            // OTD3.3 — one particle's card from the Standard Model deck.
            // Evaluated with the deferred sims so the console reads in
            // the order the program was written.
            Stmt::Particle(name) => {
                let mut lines = crate::world::particles::particle_card(&name);
                w.console.append(&mut lines);
            }
            _ => {}
        }
    }
    w
}

fn scan_unit(src: &str) -> Option<String> {
    for line in src.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix("unit") {
            let rest = rest.trim_start();
            let rest = rest.strip_prefix(':').or_else(|| rest.strip_prefix(' ')).unwrap_or(rest);
            let rest = rest.trim();
            if crate::lang::keywords::UNITS.contains(&rest) {
                return Some(rest.to_string());
            }
        }
    }
    None
}

fn finalize_stats(w: &mut World) {
    let mut vol = 0.0;
    let mut mass = 0.0;
    let mut area = 0.0;
    let mut com_num = V3::ZERO;
    let mut visible = 0usize;
    let mut bb = Aabb::empty();
    for p in &w.parts {
        if p.hidden {
            continue;
        }
        visible += 1;
        vol += p.volume_mm3;
        mass += p.mass_g;
        area += p.area_mm2;
        if let Some(c) = p.centroid {
            com_num = com_num.add(&c.mul(p.volume_mm3));
        }
        if !p.mesh.is_empty() {
            bb.grow_box(&p.mesh.bbox());
        }
    }
    let com = if vol > 1e-9 { Some(com_num.mul(1.0 / vol)) } else { None };
    w.stats = Stats {
        total_volume_mm3: vol,
        total_mass_g: mass,
        total_area_mm2: area,
        com,
        bbox: bb,
        visible_parts: visible,
        overlaps: false,
        avg_density_kg_m3: if vol > 1e-9 { (mass / 1000.0) / (vol * 1e-9) } else { 0.0 },
    };
    // overlap warning (mass approximate) — tolerance-based so resting
    // contact (touching boxes) after settle is NOT a false alarm
    // fluids (liquids + gases) are exempt: they interpenetrate BY NATURE —
    // that is what mixing means. Solidity is a law about SOLIDS.
    // OTD4: when strict_overlap is on, this becomes a hard error.
    use super::materials as mats;
    let is_fluid = |p: &Part| p.material.map(|m| mats::state(m)) != Some(mats::State::Solid);
    let boxes: Vec<(Aabb, bool, String)> = w.parts.iter().filter(|p| !p.hidden && !p.mesh.is_empty()).map(|p| (p.mesh.bbox(), is_fluid(p), p.name.clone())).collect();
    let mut first_overlap: Option<(String, String, f64)> = None;
    'outer: for i in 0..boxes.len() {
        for j in i + 1..boxes.len() {
            if boxes[i].1 || boxes[j].1 { continue; }
            let a = &boxes[i].0;
            let b = &boxes[j].0;
            let pen3 = |k: usize| a.max.0[k].min(b.max.0[k]) - a.min.0[k].max(b.min.0[k]);
            if pen3(0) > 0.02 && pen3(1) > 0.02 && pen3(2) > 0.02 {
                let depth = pen3(0).min(pen3(1)).min(pen3(2));
                first_overlap = Some((boxes[i].2.clone(), boxes[j].2.clone(), depth));
                w.stats.overlaps = true;
                break 'outer;
            }
        }
    }
    if let Some((a, b, depth)) = first_overlap {
        if w.strict_overlap {
            // Hard error — the silent-wrongness fix.
            w.errors.push(Error::new(0, format!("strict overlap: {} and {} interpenetrate by {:.2} mm — compilation FAILED", a, b, depth))
                .with_hint(format!("fix: separate them (move along the shallowest axis by {:.2} mm), or fuse with `add {}, {}` if they are meant to be one part", depth, a, b)));
            w.console.push(ConsoleLine {
                kind: LineKind::Error,
                text: format!("strict overlap: {} and {} interpenetrate by {:.2} mm — use `strict: off` to demote to warning, or fix the geometry", a, b, depth),
            });
        } else {
            w.console.push(ConsoleLine {
                kind: LineKind::Warn,
                text: format!("objects {} and {} overlap by {:.2} mm — mass counts the overlap twice until you fuse them with add (run `strict: overlap` to make this an error, or `simulate: settle` / `simulate: solidity` to fix/check)", a, b, depth),
            });
        }
    }

    // OTD4 — propagate magnetize: marks from the world registry onto parts.
    for p in w.parts.iter_mut() {
        if w.magnet_moments.contains_key(&p.name) {
            p.magnetized = true;
        }
    }
}

impl<'a> Ctx<'a> {
    fn exec(&mut self, stmt: Stmt) -> Flow {
        match stmt {
            Stmt::Scene(s) => self.world.title = s,
            Stmt::Version(v) => {
                self.world.version = Some(v);
                if v > 2.0 {
                    self.world.console.push(ConsoleLine {
                        kind: LineKind::Warn,
                        text: format!("this file asks for OTD {} — running it as OTD 2.1 (newer features may be missing)", v),
                    });
                }
            }
            Stmt::Unit(_) => {} // handled in pre-scan
            Stmt::Gravity(g) => {
                self.world.gravity = match g.as_str() {
                    "off" => 0.0,
                    "moon" => crate::units::G_MOON,
                    "mars" => crate::units::G_MARS,
                    "earth" => crate::units::G_EARTH,
                    other => other.parse().unwrap_or(crate::units::G_EARTH),
                };
            }
            Stmt::Environment(v) => {
                self.world.environment = v;
            }
            Stmt::Temperature(spec) => {
                // accepts: 300 (°C implied), 350K, 72F, -40, 25C
                let line = 0usize;
                let s = spec.trim();
                let (num_part, unit_part) = s.split_at(
                    s.find(|c: char| c.is_alphabetic()).unwrap_or(s.len())
                );
                let v: f64 = match num_part.trim().parse() {
                    Ok(v) => v,
                    Err(_) => {
                        self.world.console.push(ConsoleLine {
                            kind: LineKind::Error,
                            text: format!("line {}: temperature needs a number like temperature: 800, 350K, or 72F", line),
                        });
                        return Flow::Continue;
                    }
                };
                let c = match unit_part.trim().to_uppercase().as_str() {
                    "" | "C" | "°C" | "CELSIUS" => v,
                    "K" | "KELVIN" => v - 273.15,
                    "F" | "FAHRENHEIT" => (v - 32.0) * 5.0 / 9.0,
                    other => {
                        self.world.console.push(ConsoleLine {
                            kind: LineKind::Error,
                            text: format!("line {}: '{}' is not a temperature unit — use C, K, or F", line, other),
                        });
                        return Flow::Continue;
                    }
                };
                self.world.temp_c = c;
            }
            Stmt::Mix { a, b, line } => {
                let lines = super::chem::mix_pair(&a, &b, line);
                self.world.console.extend(lines);
            }
            // OTD4 — multi-part assembly: include "parts/wheel.otd" at (x,y,z)
            Stmt::Include { file, at, line } => {
                self.exec_include(file, at, line);
            }
            // OTD4 — magnetize: mark a part as a permanent magnet
            Stmt::Magnetize { target, moment, line } => {
                self.exec_magnetize(target, moment, line);
            }
            // OTD4 — strict: overlap | all | off — toggle strict-mode flags
            Stmt::Strict(mode) => {
                self.world.strict_overlap = matches!(mode.as_str(), "overlap" | "all" | "on" | "yes");
                self.world.console.push(ConsoleLine {
                    kind: LineKind::Info,
                    text: format!(
                        "strict mode: {} — {}",
                        mode,
                        if self.world.strict_overlap {
                            "solid-solid interpenetration will FAIL compilation (not just warn)"
                        } else {
                            "silent-wrongness back to warnings only"
                        }
                    ),
                });
            }
            // OTD4 — overlap: check — explicit overlap audit
            Stmt::Overlap { mode, line } => {
                let lines = self.overlap_audit(line, mode == "strict");
                self.world.console.extend(lines);
            }
            // OTD6 #5: connect: A B — record electrical connectivity
            Stmt::Connect { a, b, line: _ } => {
                self.world.connections.push((a.clone(), b.clone()));
                self.world.console.push(ConsoleLine {
                    kind: LineKind::Info,
                    text: format!("connect: {} ↔ {} — electrical path established (run simulate: circuit to analyze)", a, b),
                });
            }
            Stmt::Camera(v) => self.world.camera = Some(v),
            Stmt::Hide(name) => self.set_hidden(&name, true),
            Stmt::Show(name) => self.set_hidden(&name, false),
            Stmt::Define { name, params, body, line } => {
                self.templates.insert(name, Template { params, body: Some(body), stmts: Vec::new(), line });
            }
            Stmt::DefineBlock { name, params, stmts, line } => {
                self.templates.insert(name, Template { params, body: None, stmts, line });
            }
            Stmt::Use { name, args, mods, line } => {
                if let Ok(Val::Shape(mut sv)) = self.eval_template(&name, &args, line) {
                    for m in &mods {
                        self.apply_mod(&mut sv, m, line);
                    }
                    self.entries.push(Entry { name: name.clone(), sv, hidden: false });
                }
            }
            Stmt::Assign(name, expr, line) => {
                match self.eval_expr(&expr, line) {
                    Ok(Val::Shape(sv)) => {
                        self.env_insert(name.clone(), Val::Shape(sv.clone()));
                        if self.tpl_mode {
                            // inside define … end: locals never reach the stage
                            self.tpl_val = Some(Val::Shape(sv));
                        } else {
                            self.entries.push(Entry { name: name.clone(), sv, hidden: false });
                        }
                    }
                    Ok(v) => {
                        if self.tpl_mode {
                            self.tpl_val = Some(v.clone());
                        }
                        self.env_insert(name, v);
                    }
                    Err(_) => {} // error already recorded
                }
            }
            Stmt::AssignOp { name, op, val, line } => self.exec_assign_op(name, op, val, line),
            Stmt::Add { first, rest, line } => self.exec_add(first, rest, line),
            Stmt::Apply { kind, target, value, line } => self.exec_apply(kind, target, value, line),
            Stmt::Export { fmt, file, line } => {
                self.world.exports.push((fmt.clone(), file.clone()));
                self.world.console.push(ConsoleLine {
                    kind: LineKind::Info,
                    text: format!("export {} → \"{}\" (use the Export button or otd --export)", fmt, file),
                });
                let _ = line;
            }
            Stmt::Print(s, line) => {
                let text = self.interpolate(&s, line);
                self.world.console.push(ConsoleLine { kind: LineKind::Print, text });
            }
            Stmt::ExprStmt(e, line) => {
                if let Ok(Val::Shape(sv)) = self.eval_expr(&e, line) {
                    if self.tpl_mode {
                        self.tpl_val = Some(Val::Shape(sv.clone()));
                    } else {
                        let name = format!("object {}", self.entries.len() + 1);
                        self.entries.push(Entry { name, sv, hidden: false });
                    }
                }
            }
            // ---- 2.1: control flow ----
            Stmt::If { arms, els, line } => return self.exec_if(arms, els, line),
            Stmt::For { var, from, to, step, body, line } => return self.exec_for(var, from, to, step, body, line),
            Stmt::ForIn { var, iter, body, line } => return self.exec_for_in(var, iter, body, line),
            Stmt::While { cond, body, line } => return self.exec_while(cond, body, line),
            Stmt::Break(line) => {
                if self.in_loop == 0 {
                    self.err_hint(line, "break only works inside for or while", "it leaves the innermost loop early");
                } else {
                    return Flow::Break;
                }
            }
            Stmt::Continue(line) => {
                if self.in_loop == 0 {
                    self.err_hint(line, "continue only works inside for or while", "it skips to the next pass of the loop");
                } else {
                    return Flow::Continue;
                }
            }
            Stmt::Assert { cond, msg, line } => {
                match self.eval_expr(&cond, line).as_ref().map_err(|e| e.clone()).and_then(|v| self.truthy(v, line)) {
                    Ok(true) => {}
                    Ok(false) => {
                        let m = msg.unwrap_or_else(|| "you said this must be true, but it isn't".into());
                        self.err_hint(line, format!("assert failed: {m}"), "change the condition, or fix the values it checks");
                    }
                    Err(_) => {} // error already recorded
                }
            }
            Stmt::Simulate(..) | Stmt::Ask(..) | Stmt::Particle(..) => unreachable!("deferred"),
        }
        Flow::Normal
    }

    fn exec_if(&mut self, arms: Vec<(Expr, Vec<Stmt>)>, els: Option<Vec<Stmt>>, line: usize) -> Flow {
        for (cond, body) in &arms {
            let hit = match self.eval_expr(cond, line) {
                Ok(v) => self.truthy(&v, line).unwrap_or(false),
                Err(_) => false,
            };
            if hit {
                return self.exec_block_inner(body);
            }
        }
        if let Some(body) = &els {
            return self.exec_block_inner(body);
        }
        Flow::Normal
    }

    /// Like exec_block but does NOT raise the loop depth: an if inside a
    /// loop body must let break/continue pass through to the loop.
    fn exec_block_inner(&mut self, stmts: &[Stmt]) -> Flow {
        let mut flow = Flow::Normal;
        for s in stmts {
            flow = self.exec(s.clone());
            if flow != Flow::Normal {
                break;
            }
        }
        flow
    }

    fn exec_for(&mut self, var: String, from: Expr, to: Expr, step: Option<Expr>, body: Vec<Stmt>, line: usize) -> Flow {
        let from = match self.eval_expr(&from, line) {
            Ok(Val::Qty(q)) => q,
            _ => {
                self.err_hint(line, "for counts numbers — like for i = 1 to 10", "or step through lengths: for x = 0cm to 10cm by 1cm");
                return Flow::Normal;
            }
        };
        let to = match self.eval_expr(&to, line) {
            Ok(Val::Qty(q)) => q,
            _ => {
                self.err_hint(line, "for counts up to a number — like for i = 1 to 10", "the value after 'to' must be a number");
                return Flow::Normal;
            }
        };
        // default step: +1 (auto-reverse when counting down); lengths step 1mm, angles 1deg
        let default_step = match from.dim {
            Dim::Length => 1.0,
            Dim::Angle => 1.0,
            _ => 1.0,
        };
        let (step, step_dim_ok) = match step {
            Some(e) => match self.eval_expr(&e, line) {
                Ok(Val::Qty(q)) => (q.v, true),
                _ => (1.0, false),
            },
            None => (if from.v <= to.v { default_step } else { -default_step }, true),
        };
        if !step_dim_ok {
            self.err_hint(line, "by wants a number — like for i = 1 to 10 by 2", "the step is how much i grows each pass");
        }
        if step == 0.0 {
            self.err_hint(line, "a for step of zero never moves", "try by 1 or by -1");
            return Flow::Normal;
        }
        let dim = from.dim;
        let mut v = from.v;
        let up = step > 0.0;
        let saved = self.env.get(&var).cloned();
        self.in_loop += 1;
        let mut flow = Flow::Normal;
        let mut iters = 0u64;
        while if up { v <= to.v + 1e-9 } else { v >= to.v - 1e-9 } {
            iters += 1;
            if iters > 100_000 {
                self.err_hint(line, "this for would run forever", "check the 'to' and 'by' values — they must eventually meet");
                break;
            }
            self.env.insert(var.clone(), Val::Qty(Qty { v, dim }));
            self.vm_fresh();
            for s in &body {
                flow = self.exec(s.clone());
                if flow != Flow::Normal { break; }
            }
            if flow == Flow::Break {
                flow = Flow::Normal;
                break;
            }
            if flow == Flow::Continue {
                flow = Flow::Normal;
            }
            v += step;
        }
        self.in_loop -= 1;
        self.restore(var, saved);
        flow
    }

    fn exec_for_in(&mut self, var: String, iter: Expr, body: Vec<Stmt>, line: usize) -> Flow {
        let items: Vec<Val> = match self.eval_expr(&iter, line) {
            Ok(Val::List(vs)) => vs,
            Ok(Val::Tuple(qs)) => qs.into_iter().map(Val::Qty).collect(),
            Ok(Val::Range(lo, hi)) => {
                let n = (hi - lo + 1.0).round();
                if n < 0.0 || n > 100_000.0 {
                    self.err_hint(line, "that range is too big to walk through", "ranges count things — keep them under 100,000");
                    return Flow::Normal;
                }
                (lo as i64..=hi as i64).map(|x| Val::Qty(Qty::plain(x as f64))).collect()
            }
            Ok(Val::Str(s)) => s.chars().map(|c| Val::Str(c.to_string())).collect(),
            Ok(other) => {
                self.err_hint(line, format!("for … in walks a list or a range, not a {}", val_name(&other)), "like for x in [1, 2, 3] or for i in 1..10");
                return Flow::Normal;
            }
            Err(_) => return Flow::Normal,
        };
        let saved = self.env.get(&var).cloned();
        self.in_loop += 1;
        let mut flow = Flow::Normal;
        for item in items {
            self.env.insert(var.clone(), item);
            self.vm_fresh();
            for s in &body {
                flow = self.exec(s.clone());
                if flow != Flow::Normal { break; }
            }
            if flow == Flow::Break {
                flow = Flow::Normal;
                break;
            }
            if flow == Flow::Continue {
                flow = Flow::Normal;
            }
        }
        self.in_loop -= 1;
        self.restore(var, saved);
        flow
    }

    fn exec_while(&mut self, cond: Expr, body: Vec<Stmt>, line: usize) -> Flow {
        let mut flow = Flow::Normal;
        self.in_loop += 1;
        let mut iters = 0u64;
        loop {
            let keep = match self.eval_expr(&cond, line) {
                Ok(v) => self.truthy(&v, line).unwrap_or(false),
                Err(_) => false,
            };
            if !keep { break; }
            iters += 1;
            if iters > 100_000 {
                self.err_hint(line, "this while never ends", "does something inside change the condition? add it, or use break");
                break;
            }
            self.vm_fresh();
            for s in &body {
                flow = self.exec(s.clone());
                if flow != Flow::Normal { break; }
            }
            if flow == Flow::Break {
                flow = Flow::Normal;
                break;
            }
            if flow == Flow::Continue {
                flow = Flow::Normal;
            }
        }
        self.in_loop -= 1;
        flow
    }

    fn restore(&mut self, name: String, old: Option<Val>) {
        match old {
            Some(v) => { self.env.insert(name, v); }
            None => { self.env.remove(&name); }
        }
    }

    /// P0650: the VM cache is keyed by AST node address. Loop bodies are
    /// re-cloned every pass, so freed addresses recycle — clear the cache
    /// each iteration or a shape call can collide with a stale program.
    /// (Pattern bodies repeat inside one statement and keep their cache.)
    fn vm_fresh(&mut self) {
        self.vm_cache.clear();
        self.vm_duds.clear();
    }

    /// Journaling env write: inside a template body, every first write to a
    /// name is remembered so it can be undone when the call returns.
    fn env_insert(&mut self, k: String, v: Val) {
        if let Some(j) = self.journal.as_mut() {
            if !j.iter().any(|(jk, _)| *jk == k) {
                let old = self.env.get(&k).cloned();
                j.push((k.clone(), old));
            }
        }
        self.env.insert(k, v);
    }

    fn exec_assign_op(&mut self, name: String, op: BinOp, val: Expr, line: usize) {
        let cur = match self.env.get(&name).cloned() {
            Some(v) => v,
            None => {
                self.err_hint(
                    line,
                    format!("make {name} first"),
                    format!("like {name} = 0, then {name} += 5"),
                );
                return;
            }
        };
        let rhs = match self.eval_expr(&val, line) {
            Ok(v) => v,
            Err(_) => return,
        };
        match self.binop_vals(op, cur, rhs, line) {
            Ok(Val::Shape(sv)) => {
                self.env_insert(name.clone(), Val::Shape(sv.clone()));
                if self.tpl_mode {
                    self.tpl_val = Some(Val::Shape(sv.clone()));
                } else if let Some(e) = self.entries.iter_mut().find(|e| e.name == name) {
                    e.sv = sv; // cup += handle grows the cup already on stage
                } else {
                    self.entries.push(Entry { name: name.clone(), sv, hidden: false });
                }
            }
            Ok(v) => self.env_insert(name, v),
            Err(_) => {} // error already recorded
        }
    }

    fn set_hidden(&mut self, name: &str, hidden: bool) {
        for e in self.entries.iter_mut() {
            if e.name == name {
                e.hidden = hidden;
            }
        }
    }

    // ---------- OTD4: include, magnetize, overlap audit ----------

    /// `include: "parts/wheel.otd" at (x, y, z)` — load another .otd file as a
    /// part library and merge its top-level shapes into this scene. This is
    /// the multi-part design workflow: make each part in its own file, then a
    /// main file `include`s them and arranges them with `at`.
    fn exec_include(&mut self, file: String, at: Option<Vec<Expr>>, line: usize) {
        // resolve relative to cwd or examples/
        let mut path = std::path::PathBuf::from(&file);
        if !path.exists() {
            path = std::path::PathBuf::from("examples").join(&file);
        }
        if !path.exists() {
            path = std::path::PathBuf::from("library/parts").join(&file);
        }
        let src = match std::fs::read_to_string(&path) {
            Ok(s) => s,
            Err(_) => {
                self.err_hint(
                    line,
                    format!("include: I can't find the file \"{}\"", file),
                    "put it next to your main .otd file, in examples/, or in library/parts/",
                );
                return;
            }
        };
        // pre-scan the file's default unit so bare numbers parse correctly
        let default_unit = scan_unit(&src).unwrap_or_else(|| self.default_unit.clone());
        let prog = parser::parse(&src, &default_unit);
        // propagate any parse errors with the include line
        for e in prog.errors {
            let mapped = Error::new(line, format!("include \"{}\": {}", file, e.msg))
                .with_hint(e.hint.unwrap_or_default());
            self.errors.push(mapped);
        }
        // count top-level shape assignments (any Stmt::Assign where the value
        // is or contains a shape); execute them in a child context that
        // shares our world but a fresh env, then promote each entry into ours
        let mut n_added = 0usize;
        // save our env/templates, run the file in a fresh sub-env
        let saved_env = std::mem::take(&mut self.env);
        let saved_templates = std::mem::take(&mut self.templates);
        let saved_magic = std::mem::take(&mut self.magic);
        let saved_entries_len = self.entries.len();
        for stmt in prog.stmts {
            let _ = self.exec(stmt);
        }
        // promote the new entries (with optional offset)
        let mut offset = V3::ZERO;
        if let Some(items) = at {
            let mut pos = [0.0f64; 3];
            for (k, item) in items.iter().take(3).enumerate() {
                if let Ok(Val::Qty(q)) = self.eval_expr(item, line) {
                    pos[k] = self.to_len(&q);
                }
            }
            if items.len() == 2 {
                pos = [pos[0], 0.0, pos[1]];
            }
            offset = V3::new(pos[0], pos[1], pos[2]);
        }
        // give each new entry a unique name based on the file name + its index
        let base_name = std::path::Path::new(&file)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("part")
            .to_string();
        let mut promoted = self.entries.split_off(saved_entries_len);
        // Count how many existing entries already use the base_name or base_name_N
        // pattern, so we can pick fresh unique names for the new ones.
        let existing_count = self.entries.iter().filter(|x| {
            x.name == base_name || x.name.starts_with(&format!("{}_", base_name))
        }).count();
        for (i, e) in promoted.iter_mut().enumerate() {
            // Always rename included parts to base_name_N to avoid clashes
            // when the same file is included multiple times.
            e.name = format!("{}_{}", base_name, existing_count + i + 1);
            if offset != V3::ZERO {
                e.sv.translate(offset);
            }
            n_added += 1;
        }
        self.entries.append(&mut promoted);
        // restore our env/templates/magic
        self.env = saved_env;
        self.templates = saved_templates;
        self.magic = saved_magic;
        self.world.console.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("included {} — {} part(s) merged into the scene{}", file, n_added, if offset != V3::ZERO { format!(" at offset ({:.0}, {:.0}, {:.0}) mm", offset.x(), offset.y(), offset.z()) } else { String::new() }),
        });
    }

    /// `magnetize: rotor [moment: 0.5]` — mark a part as a permanent magnet.
    fn exec_magnetize(&mut self, target: String, moment: Option<Expr>, line: usize) {
        // validate that the named part exists (in entries)
        let exists = self.entries.iter().any(|e| e.name == target);
        if !exists {
            let suggestion = self.entries.iter().map(|e| e.name.as_str()).collect::<Vec<_>>();
            let hint = crate::lang::errors::suggest(&target, &suggestion)
                .map(|s| format!("did you mean {}?", s))
                .unwrap_or_else(|| "make the part first, then magnetize it".to_string());
            self.err_hint(line, format!("magnetize: no part named '{}'", target), hint);
            return;
        }
        // compute moment (A·m²): explicit, or estimate from the part's volume
        let m_am2 = if let Some(expr) = moment {
            match self.eval_expr(&expr, line) {
                Ok(Val::Qty(q)) => q.expect_plain("magnetic moment"),
                _ => {
                    self.err_hint(line, "magnetize moment: needs a number", "like magnetize: rotor moment: 0.5  (units: A·m²)");
                    return;
                }
            }
        } else {
            // estimate from the part's mesh volume: a neodymium magnet
            // carries ~8×10⁵ A/m magnetisation; compute m = M × V
            let entry = self.entries.iter().find(|e| e.name == target).unwrap();
            let vol_m3 = entry.sv.bbox().size().x() * entry.sv.bbox().size().y() * entry.sv.bbox().size().z() * 1e-9;
            8.0e5 * vol_m3
        };
        // store on the world's magnet registry — applied to parts after finalize
        self.world.magnet_moments.insert(target.clone(), m_am2);
        self.world.console.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("magnetize: {} is now a permanent magnet, moment ≈ {:.3} A·m² (run simulate: magnet to see its field)", target, m_am2),
        });
    }

    /// `overlap: check` — explicit overlap audit. When `strict` is true,
    /// overlapping pairs become hard errors instead of warnings.
    fn overlap_audit(&mut self, line: usize, force_strict: bool) -> Vec<ConsoleLine> {
        let mut out = Vec::new();
        let strict = self.world.strict_overlap || force_strict;
        // build pairs from current entries (their AABBs)
        use super::materials as mats;
        let is_fluid = |e: &Entry| e.sv.mat.map(|m| mats::state(m)) != Some(mats::State::Solid);
        let boxes: Vec<(usize, String, Aabb, bool)> = self.entries.iter().enumerate()
            .filter(|(_, e)| !e.hidden && !e.sv.is_empty())
            .map(|(i, e)| (i, e.name.clone(), e.sv.bbox(), is_fluid(e)))
            .collect();
        let mut bad_pairs: Vec<(String, String, f64, usize)> = Vec::new();
        for i in 0..boxes.len() {
            for j in i + 1..boxes.len() {
                if boxes[i].3 || boxes[j].3 {
                    continue; // fluids interpenetrate by nature
                }
                let a = &boxes[i].2;
                let b = &boxes[j].2;
                let pen = |k: usize| a.max.0[k].min(b.max.0[k]) - a.min.0[k].max(b.min.0[k]);
                if pen(0) > 0.02 && pen(1) > 0.02 && pen(2) > 0.02 {
                    let depth = pen(0).min(pen(1)).min(pen(2));
                    let axis = [pen(0), pen(1), pen(2)].iter().enumerate().min_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
                    bad_pairs.push((boxes[i].1.clone(), boxes[j].1.clone(), depth, axis));
                }
            }
        }
        if bad_pairs.is_empty() {
            out.push(ConsoleLine {
                kind: LineKind::Answer,
                text: format!("OVERLAP CHECK: SOLID — {} solid part(s), zero interpenetrations", boxes.len()),
            });
        } else {
            out.push(ConsoleLine {
                kind: if strict { LineKind::Error } else { LineKind::Warn },
                text: format!("OVERLAP CHECK: {} overlapping pair(s)", bad_pairs.len()),
            });
            for (a, b, depth, axis) in &bad_pairs {
                let axis_name = ["X", "Y", "Z"][*axis];
                let suggestion = format!(
                    "fix: move {} along {} by {:.2} mm, or fuse them with `add {}, {}` if they are meant to be one part",
                    b, axis_name, depth, a, b
                );
                out.push(ConsoleLine {
                    kind: if strict { LineKind::Error } else { LineKind::Warn },
                    text: format!("  {} and {} overlap by {:.2} mm (shallowest axis: {}) — {}", a, b, depth, axis_name, suggestion),
                });
                if strict {
                    self.err_hint(line, format!("{} and {} interpenetrate by {:.2} mm", a, b, depth), suggestion);
                }
            }
        }
        out
    }

    /// `print "the radius is {r}"` — {name} pulls the value from the
    /// environment (2.1). Unknown names stay as written.
    fn interpolate(&mut self, s: &str, line: usize) -> String {
        if !s.contains('{') {
            return s.to_string();
        }
        let mut out = String::new();
        let mut rest = s;
        while let Some(open) = rest.find('{') {
            out.push_str(&rest[..open]);
            let after = &rest[open + 1..];
            match after.find('}') {
                Some(close) => {
                    let expr_str = after[..close].trim();
                    // OTD6 #6: expression interpolation — try bare name first,
                    // then fall back to evaluating it as an expression (a+b, sqrt(x), etc.)
                    let val = self
                        .env
                        .get(expr_str)
                        .cloned()
                        .or_else(|| self.magic.iter().find(|(k, _)| k == expr_str).map(|(_, q)| Val::Qty(*q)));
                    match val {
                        Some(v) => out.push_str(&format_val(&v)),
                        None => {
                            // OTD6 #6: try evaluating as an expression (not just a bare name)
                            // Parse the expression and evaluate it. If parsing or
                            // evaluation fails, silently fall through to the warning
                            // (don't push errors — this is a print interpolation, not
                            // a real statement).
                            let parsed = crate::lang::parser::parse(&format!("__tmp = {}", expr_str), &self.default_unit);
                            if parsed.errors.is_empty() && parsed.stmts.len() == 1 {
                                if let crate::lang::ast::Stmt::Assign(_, e, _) = &parsed.stmts[0] {
                                    let err_count_before = self.errors.len();
                                    let eval_result = self.eval_expr(e, line);
                                    if eval_result.is_err() {
                                        // roll back any errors pushed during eval
                                        self.errors.truncate(err_count_before);
                                    } else if let Ok(v) = eval_result {
                                        out.push_str(&format_val(&v));
                                        rest = &after[close + 1..];
                                        continue;
                                    }
                                }
                            }
                            // Expression evaluation failed — warn and leave as literal
                            self.world.console.push(ConsoleLine {
                                kind: LineKind::Warn,
                                text: format!(
                                    "line {}: print interpolation '{}' is not a known variable or expression — left as literal text. Define it first (r = 5cm) or fix the typo.",
                                    line, expr_str
                                ),
                            });
                            out.push('{');
                            out.push_str(expr_str);
                            out.push('}');
                        }
                    }
                    rest = &after[close + 1..];
                }
                None => {
                    out.push('{');
                    out.push_str(after);
                    rest = "";
                }
            }
        }
        out.push_str(rest);
        out
    }

    fn exec_add(&mut self, first: Expr, rest: Vec<Expr>, line: usize) {
        // target: an existing entry (by name) or a fresh shape
        let first_name = match &first {
            Expr::Ident(n) => Some(n.clone()),
            _ => None,
        };
        let mut target_sv = match self.eval_expr(&first, line) {
            Ok(Val::Shape(sv)) => sv,
            _ => {
                self.err_hint(line, "add starts with an object you already made, like add cup, handle", "the first name is the one that keeps its identity");
                return;
            }
        };
        for r in rest {
            // remove the entry if the operand is a plain name
            let r_name = match &r {
                Expr::Ident(n) => Some(n.clone()),
                _ => None,
            };
            match self.eval_expr(&r, line) {
                Ok(Val::Shape(mut sv)) => {
                    // union into target
                    let mut merged = target_sv.merged_mesh();
                    for m in sv.meshes.drain(..) {
                        merged = mesh_union(&merged, &m);
                    }
                    target_sv = ShapeVal {
                        meshes: vec![merged],
                        kinds: vec![Kind::Generic],
                        mat: target_sv.mat.or(sv.mat),
                        color: target_sv.color.or(sv.color),
                    };
                }
                _ => {
                    self.err_hint(line, "add only fuses objects", "did you forget to make this one first?");
                }
            }
            if let Some(rn) = r_name {
                self.entries.retain(|e| e.name != rn);
                self.env.remove(&rn);
            }
        }
        match first_name {
            Some(n) => {
                // replace existing entry or add new
                if let Some(e) = self.entries.iter_mut().find(|e| e.name == n) {
                    e.sv = target_sv;
                } else {
                    self.entries.push(Entry { name: n, sv: target_sv, hidden: false });
                }
            }
            None => {
                let name = format!("object {}", self.entries.len() + 1);
                self.entries.push(Entry { name, sv: target_sv, hidden: false });
            }
        }
    }

    fn exec_apply(&mut self, kind: ApplyKind, target: Option<String>, value: String, line: usize) {
        match kind {
            ApplyKind::Material => {
                let m = match materials::find(&value) {
                    Some(m) => m,
                    None => {
                        let hint = materials::suggest(&value)
                            .map(|s| format!("did you mean {}?", s))
                            .unwrap_or_else(|| format!("materials: {}… (see the cheat sheet)", materials::NAMES[..3].concat()));
                        self.err_hint(line, format!("'{}' is not a material I know", value), hint);
                        return;
                    }
                };
                match target {
                    Some(t) => {
                        let mut found = false;
                        for e in self.entries.iter_mut() {
                            if e.name == t {
                                e.sv.mat = Some(m);
                                found = true;
                            }
                        }
                        if !found {
                            self.err_hint(line, format!("I can't find an object named '{}'", t), "material name: material cup: ceramic");
                        }
                    }
                    None => {
                        self.world.default_material = Some(m);
                        // retroactively: later defaults fill un-set parts (finalize step)
                    }
                }
            }
            ApplyKind::Color => {
                let c = match colors::resolve(&value) {
                    Some(c) => c,
                    None => {
                        let hint = colors::suggest(&value)
                            .map(|s| format!("did you mean {}?", s))
                            .unwrap_or_else(|| "colors are words like ivory or steelblue, or #rrggbb".into());
                        self.err_hint(line, format!("'{}' is not a color I know", value), hint);
                        return;
                    }
                };
                match target {
                    Some(t) => {
                        let mut found = false;
                        for e in self.entries.iter_mut() {
                            if e.name == t {
                                e.sv.color = Some(c);
                                found = true;
                            }
                        }
                        if !found {
                            self.err_hint(line, format!("I can't find an object named '{}'", t), "color name: color cup: ivory");
                        }
                    }
                    None => self.world.default_color = Some(c),
                }
            }
        }
    }

    pub(super) fn plain_err(&self, line: usize, msg: impl Into<String>) -> Error {
        Error::new(line, msg)
    }

    /// Value as millimeters. Explicit units pass through; bare numbers get
    /// the scene default unit (cm by default).
    pub(super) fn to_len(&self, q: &Qty) -> f64 {
        match q.dim {
            Dim::Length => q.v,
            Dim::Angle => q.v, // caller reports the mismatch
            Dim::Plain => q.v * self.unit_mm,
        }
    }

    // ---------- expression evaluation ----------

    pub(super) fn eval_expr(&mut self, e: &Expr, line: usize) -> Result<Val, Error> {
        // P0650: the virtual machine gets first crack at numeric expressions
        if matches!(e, Expr::Bin(..) | Expr::Neg(..) | Expr::Call { .. } | Expr::Ident(_)) {
            if let Some(res) = self.try_vm(e, line) {
                return res;
            }
        }
        match e {
            Expr::Num(q) => Ok(Val::Qty(*q)),
            Expr::Str(s) => Ok(Val::Str(s.clone())),
            Expr::Bool(b) => Ok(Val::Bool(*b)),
            Expr::Ident(name) => self.eval_ident(name, line),
            Expr::Tuple(items) => {
                let mut qs = Vec::new();
                for i in items {
                    match self.eval_expr(i, line)? {
                        Val::Qty(q) => qs.push(q),
                        _ => return Err(self.err_hint(line, "a tuple holds numbers, like (4cm, 5cm, 0)", "positions are (x, y, z)")),
                    }
                }
                Ok(Val::Tuple(qs))
            }
            Expr::List(items) => {
                let mut vs = Vec::new();
                for i in items {
                    vs.push(self.eval_expr(i, line)?);
                }
                Ok(Val::List(vs))
            }
            Expr::Neg(inner) => match self.eval_expr(inner, line)? {
                Val::Qty(q) => Ok(Val::Qty(Qty { v: -q.v, dim: q.dim })),
                other => Err(self.err_hint(line, "you can only flip the sign of a number", format!("got {}", val_name(&other)))),
            },
            Expr::Not(inner) => {
                let v = self.eval_expr(inner, line)?;
                let b = self.truthy(&v, line)?;
                Ok(Val::Bool(!b))
            }
            Expr::Range(lo, hi) => {
                let l = self.eval_expr(lo, line)?;
                let h = self.eval_expr(hi, line)?;
                match (l, h) {
                    (Val::Qty(a), Val::Qty(b)) if a.dim == Dim::Plain && b.dim == Dim::Plain => {
                        Ok(Val::Range(a.v.round(), b.v.round()))
                    }
                    (Val::Qty(q), _) if q.dim != Dim::Plain => {
                        Err(self.err_hint(line, "ranges count things — plain numbers only", "to step through lengths use for x = 0cm to 10cm by 1cm"))
                    }
                    _ => Err(self.err_hint(line, "a range is two plain numbers — like 1..12", "for i in 1..12")),
                }
            }
            Expr::Index(base, idx) => self.eval_index(base, idx, line),
            Expr::Bin(BinOp::AndAlso, l, r) => {
                // short-circuit: the right side is skipped when the left decides
                let lv = self.eval_expr(l, line)?;
                if !self.truthy(&lv, line)? {
                    return Ok(Val::Bool(false));
                }
                let rv = self.eval_expr(r, line)?;
                Ok(Val::Bool(self.truthy(&rv, line)?))
            }
            Expr::Bin(BinOp::OrElse, l, r) => {
                let lv = self.eval_expr(l, line)?;
                if self.truthy(&lv, line)? {
                    return Ok(Val::Bool(true));
                }
                let rv = self.eval_expr(r, line)?;
                Ok(Val::Bool(self.truthy(&rv, line)?))
            }
            Expr::Bin(op, l, r) => self.eval_bin(*op, l, r, line),
            Expr::Postfix { base, mods } => {
                let mut v = self.eval_expr(base, line)?;
                if let Val::Shape(ref mut sv) = v {
                    for m in mods {
                        self.apply_mod(sv, m, line);
                    }
                    Ok(v)
                } else if mods.is_empty() {
                    Ok(v)
                } else {
                    Err(self.err_hint(line, "at / rotate / scale work on objects, not numbers", "make a shape first, then place it"))
                }
            }
            Expr::Call { name, args, body, .. } => self.eval_call(name, args, body.as_deref(), line),
        }
    }

    fn eval_ident(&mut self, name: &str, line: usize) -> Result<Val, Error> {
        // the one predefined constant (documented since 1.0, actually wired in 2.0)
        if name == "pi" {
            return Ok(Val::Qty(Qty::plain(std::f64::consts::PI)));
        }
        // magic vars (patterns)
        if let Some((_, q)) = self.magic.iter().find(|(k, _)| k == name) {
            return Ok(Val::Qty(*q));
        }
        // environment (assignments, params)
        if let Some(v) = self.env.get(name) {
            return Ok(v.clone());
        }
        // zero-arg template reference
        if let Some(t) = self.templates.get(name).cloned() {
            if t.params.is_empty() {
                return match t.body {
                    Some(body) => self.eval_expr(&body, t.line),
                    None => self.eval_template(name, &[], line),
                };
            }
            return Err(self.err_hint(
                line,
                format!("'{}' needs its settings, like {}({})", name, name, t.params.join(", ")),
                "templates with parameters are called like gear(r: 30mm)",
            ));
        }
        // value words
        if matches!(name, "x" | "y" | "z" | "top" | "bottom" | "none" | "front" | "back" | "left" | "right" | "on" | "off" | "yes" | "no") {
            return Ok(Val::Word(name.to_string()));
        }
        if crate::lang::keywords::is_primitive(name) {
            return Err(self.err_hint(
                line,
                format!("{} needs a size or settings", name),
                format!("like {} 5cm or {}(r: 5cm)", name, name),
            ));
        }
        if crate::lang::keywords::is_builder(name) {
            return Err(self.err_hint(
                line,
                format!("{} needs its inputs", name),
                "like extrude [(0, 0), (3cm, 0), (0, 3cm)] depth: 5mm",
            ));
        }
        if let Some(s) = crate::lang::errors::suggest(name, crate::lang::keywords::KEYWORDS) {
            return Err(self.err_hint(
                line,
                format!("I don't know what '{}' is", name),
                format!("did you mean {}?", s),
            ));
        }
        Err(self.err_hint(
            line,
            format!("I don't know what '{}' is", name),
            "did you spell it right? make it first, like cup = cylinder(height: 10cm)",
        ))
    }

    /// P0650 — the OTD-ASM path: pure-numeric expressions (units, i/j/a,
    /// pi, arithmetic, math functions) compile to 32-byte bytecode and run
    /// on the register machine. Anything else returns None and the
    /// tree-walk below handles it — semantics are identical (parity-tested
    /// + the whole 1.0 example corpus is the regression gate).
    fn try_vm(&mut self, e: &Expr, line: usize) -> Option<Result<Val, Error>> {
        use crate::vm;
        if !vm::compile::is_numeric_shape(e) {
            return None;
        }
        let key = e as *const Expr as usize;
        // compile once per AST node; remember duds so shape calls don't
        // retry compilation every pattern iteration
        if !self.vm_duds.contains(&key) && !self.vm_cache.contains_key(&key) {
            match vm::compile::compile(e, self.unit_mm) {
                Ok(p) => {
                    self.vm_cache.insert(key, p);
                }
                Err(_) => {
                    self.vm_duds.insert(key);
                    return None;
                }
            }
        }
        let prog = self.vm_cache.get(&key)?;
        // every variable must resolve numerically NOW, else the tree-walk
        // owns it (templates, shapes, value words, friendly errors)
        for name in &prog.vars {
            if self.resolve_numeric(name).is_none() {
                return None;
            }
        }
        let res = {
            let lookup = |name: &str| self.resolve_numeric(name);
            vm::interp::run(prog, &lookup)
        };
        match res {
            Ok(v) => Some(Ok(Val::Qty(v.to_qty()))),
            Err(msg) => {
                // record it: callers like `x = …` trust that an Err was
                // already pushed (2.1 audit — VM errors used to vanish there)
                let e = self.plain_err(line, msg);
                self.errors.push(e.clone());
                Some(Err(e))
            }
        }
    }

    /// Numeric variable resolution in eval_ident's order: pi, magic
    /// pattern vars, then the environment. Only Qty values count.
    fn resolve_numeric(&self, name: &str) -> Option<Qty> {
        if name == "pi" {
            return Some(Qty::plain(std::f64::consts::PI));
        }
        if let Some((_, q)) = self.magic.iter().find(|(k, _)| k == name) {
            return Some(*q);
        }
        match self.env.get(name) {
            Some(Val::Qty(q)) => Some(*q),
            _ => None,
        }
    }

    fn eval_bin(&mut self, op: BinOp, l: &Expr, r: &Expr, line: usize) -> Result<Val, Error> {
        // FIX(P1415 / B15): in `a - tool material: glass`, the trailing
        // material/color used to bind to the cutting TOOL and vanish — the
        // wooden boat in the examples rendered plastic and "sank". A drill
        // bit has no finish: hoist finish mods (material/color) off the tool
        // and paint the RESULT of the cut instead.
        let hoisted: Option<Vec<Mod>> = if matches!(op, BinOp::Sub | BinOp::And) {
            match r {
                Expr::Postfix { mods, .. } => {
                    let finish: Vec<Mod> = mods
                        .iter()
                        .filter(|m| matches!(m, Mod::Material(_) | Mod::Color(_)))
                        .cloned()
                        .collect();
                    if finish.is_empty() { None } else { Some(finish) }
                }
                _ => None,
            }
        } else {
            None
        };
        // hollow sugar: a - hollow(wall: 3mm) — also through tool mods
        // (`a - hollow(wall: 2mm) at (0, 2cm, 0) material: glass`)
        if op == BinOp::Sub {
            let hollow_call: Option<&Vec<crate::lang::ast::Arg>> = match r {
                Expr::Call { name, args, .. } if name == "hollow" => Some(args),
                Expr::Postfix { base: boxed, .. } => match boxed.as_ref() {
                    Expr::Call { name, args, .. } if name == "hollow" => Some(args),
                    _ => None,
                },
                _ => None,
            };
            if let Some(args) = hollow_call {
                let lv = self.eval_expr(l, line)?;
                if let Val::Shape(mut sv) = lv {
                    let hr = self.eval_hollow(&mut sv, args, line)?;
                    let mut out = Val::Shape(hr);
                    if let Some(finish) = &hoisted {
                        if let Val::Shape(ref mut sv2) = out {
                            for m in finish {
                                self.apply_mod(sv2, m, line);
                            }
                        }
                    }
                    return Ok(out);
                }
            }
        }
        // general path: rebind the tool WITHOUT its finish mods
        let rr: Expr = if let Some(finish) = &hoisted {
            if let Expr::Postfix { base, mods } = r {
                let kept: Vec<Mod> = mods
                    .iter()
                    .filter(|m| !matches!(m, Mod::Material(_) | Mod::Color(_)))
                    .cloned()
                    .collect();
                let _ = finish;
                Expr::Postfix { base: base.clone(), mods: kept }
            } else {
                r.clone()
            }
        } else {
            r.clone()
        };
        let lv = self.eval_expr(l, line)?;
        let rv = self.eval_expr(&rr, line)?;
        let combined = self.binop_vals(op, lv, rv, line)?;
        if let Some(finish) = &hoisted {
            let mut out = combined;
            if let Val::Shape(ref mut sv) = out {
                for m in finish {
                    self.apply_mod(sv, m, line);
                }
            }
            return Ok(out);
        }
        Ok(combined)
    }

    /// Combine two already-evaluated values with a binary operator.
    /// Shared by expressions and compound assignment (2.1).
    fn binop_vals(&mut self, op: BinOp, lv: Val, rv: Val, line: usize) -> Result<Val, Error> {
        // comparisons work on several types
        if matches!(op, BinOp::Lt | BinOp::Gt | BinOp::Le | BinOp::Ge) {
            return self.binop_cmp(op, lv, rv, line);
        }
        if matches!(op, BinOp::Eq | BinOp::Ne) {
            let eq = self.val_eq(&lv, &rv, line)?;
            return Ok(Val::Bool(if op == BinOp::Eq { eq } else { !eq }));
        }
        if op == BinOp::In {
            return self.binop_in(lv, rv, line);
        }
        match (lv, rv) {
            (Val::Shape(mut a), Val::Shape(b)) => {
                let merged = match op {
                    BinOp::Add => {
                        let mut am = a.merged_mesh();
                        for m in &b.meshes {
                            am = mesh_union(&am, m);
                        }
                        am
                    }
                    BinOp::Sub => {
                        let am = a.merged_mesh();
                        let bm = b.merged_mesh();
                        mesh_subtract(&am, &bm)
                    }
                    BinOp::And => {
                        let am = a.merged_mesh();
                        let bm = b.merged_mesh();
                        mesh_intersect(&am, &bm)
                    }
                    _ => {
                        return Err(self.err_hint(
                            line,
                            format!("objects fuse with +, cut with -, overlap with & — not {}", op.spell()),
                            "put a shape on each side, or a number",
                        ));
                    }
                };
                a.meshes = vec![merged];
                a.kinds = vec![Kind::Generic];
                Ok(Val::Shape(a))
            }
            (Val::Qty(a), Val::Qty(b)) => {
                // unit promotion for + and −: bare number + unit-ed number → the
                // unit wins (for * and / the Plain side stays a plain factor,
                // so `i * 30deg` keeps working).
                let (a, b) = if matches!(op, BinOp::Add | BinOp::Sub) {
                    self.promote_pair(a, b)
                } else {
                    (a, b)
                };
                match op {
                    BinOp::Add | BinOp::Sub => {
                        let dim = a.dim.combine_add(b.dim).map_err(|e| self.plain_err(line, e))?;
                        Ok(Val::Qty(Qty { v: if op == BinOp::Add { a.v + b.v } else { a.v - b.v }, dim }))
                    }
                    BinOp::Mul => {
                        let dim = a.dim.mul(b.dim).map_err(|e| self.plain_err(line, e))?;
                        Ok(Val::Qty(Qty { v: a.v * b.v, dim }))
                    }
                    BinOp::Div => {
                        if b.v == 0.0 {
                            return Err(self.err_hint(line, "you can't divide by zero", "check the right side of the /"));
                        }
                        let dim = a.dim.div(b.dim).map_err(|e| self.plain_err(line, e))?;
                        Ok(Val::Qty(Qty { v: a.v / b.v, dim }))
                    }
                    BinOp::Pow => {
                        if a.dim != Dim::Plain || b.dim != Dim::Plain {
                            return Err(self.err_hint(
                                line,
                                format!("{} ^ {} — powers only work on plain numbers", fmt_qty(a), fmt_qty(b)),
                                "squaring a length would make an area; OTD keeps one length dimension",
                            ));
                        }
                        Ok(Val::Qty(Qty::plain(a.v.powf(b.v))))
                    }
                    BinOp::Mod => {
                        if a.dim != Dim::Plain || b.dim != Dim::Plain {
                            return Err(self.err_hint(
                                line,
                                "mod (the remainder) works on plain numbers",
                                "like 7 mod 2 or 7 % 2",
                            ));
                        }
                        if b.v == 0.0 {
                            return Err(self.err_hint(line, "you can't take the remainder after dividing by zero", "check the right side of the mod"));
                        }
                        Ok(Val::Qty(Qty::plain(a.v % b.v)))
                    }
                    BinOp::And => Err(self.err_hint(line, "& fuses objects, not numbers", "put a shape on each side: cup & sphere 2cm")),
                    _ => unreachable!("logic and comparisons handled above"),
                }
            }
            (Val::Tuple(mut a), Val::Tuple(b)) if op == BinOp::Add && a.len() == b.len() => {
                for (x, y) in a.iter_mut().zip(b.iter()) {
                    x.v += y.v;
                }
                Ok(Val::Tuple(a))
            }
            (l, r) => Err(self.err_hint(
                line,
                format!("you can't {} a {} and a {}", op_word(op), val_name(&l), val_name(&r)),
                "shapes fuse with +, cut with -, overlap with &",
            )),
        }
    }

    /// Promote (Plain, Length|Angle) pairs for + − and comparisons: the
    /// unit-ed side wins; bare numbers get the scene default unit.
    fn promote_pair(&self, a: Qty, b: Qty) -> (Qty, Qty) {
        match (a.dim, b.dim) {
            (Dim::Plain, Dim::Length) => (Qty::mm(a.v * self.unit_mm), b),
            (Dim::Length, Dim::Plain) => (a, Qty::mm(b.v * self.unit_mm)),
            (Dim::Plain, Dim::Angle) => (Qty::deg(a.v), b),
            (Dim::Angle, Dim::Plain) => (a, Qty::deg(b.v)),
            _ => (a, b),
        }
    }

    fn binop_cmp(&mut self, op: BinOp, lv: Val, rv: Val, line: usize) -> Result<Val, Error> {
        let ord: Option<std::cmp::Ordering> = match (lv, rv) {
            (Val::Qty(a), Val::Qty(b)) => {
                let (a, b) = self.promote_pair(a, b);
                if a.dim != b.dim && a.dim != Dim::Plain && b.dim != Dim::Plain {
                    return Err(self.err_hint(
                        line,
                        format!("you can't compare {} and {} — those units don't match", fmt_qty(a), fmt_qty(b)),
                        "compare two lengths, or two angles",
                    ));
                }
                a.v.partial_cmp(&b.v)
            }
            (Val::Str(a), Val::Str(b)) => a.partial_cmp(&b),
            (l, r) => {
                return Err(self.err_hint(
                    line,
                    format!("{} compares numbers or words, not a {} and a {}", op.spell(), val_name(&l), val_name(&r)),
                    "try asking a question about a number, like x > 5cm",
                ));
            }
        };
        use std::cmp::Ordering::*;
        let b = match (op, ord) {
            (_, None) => false,
            (BinOp::Lt, Some(Less)) | (BinOp::Gt, Some(Greater))
            | (BinOp::Le, Some(Less | Equal)) | (BinOp::Ge, Some(Greater | Equal)) => true,
            _ => false,
        };
        Ok(Val::Bool(b))
    }

    /// Structural equality for == / != / in (2.1).
    fn val_eq(&mut self, l: &Val, r: &Val, line: usize) -> Result<bool, Error> {
        Ok(match (l, r) {
            (Val::Qty(a), Val::Qty(b)) => {
                let (a, b) = self.promote_pair(*a, *b);
                a.dim == b.dim && (a.v - b.v).abs() < 1e-9
            }
            (Val::Str(a), Val::Str(b)) => a == b,
            (Val::Bool(a), Val::Bool(b)) => a == b,
            (Val::Word(a), Val::Word(b)) => a == b,
            (Val::Range(a1, a2), Val::Range(b1, b2)) => a1 == b1 && a2 == b2,
            (Val::List(a), Val::List(b)) => {
                if a.len() != b.len() { return Ok(false); }
                for (x, y) in a.iter().zip(b.iter()) {
                    if !self.val_eq(x, y, line)? { return Ok(false); }
                }
                true
            }
            (Val::Tuple(a), Val::Tuple(b)) => {
                a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| x.dim == y.dim && (x.v - y.v).abs() < 1e-9)
            }
            _ => false,
        })
    }

    /// `x in xs` — membership (2.1): lists, strings, ranges.
    fn binop_in(&mut self, lv: Val, rv: Val, line: usize) -> Result<Val, Error> {
        match (lv, rv) {
            (needle, Val::List(items)) => {
                for it in &items {
                    if self.val_eq(&needle, it, line)? {
                        return Ok(Val::Bool(true));
                    }
                }
                Ok(Val::Bool(false))
            }
            (Val::Str(a), Val::Str(b)) => Ok(Val::Bool(b.contains(&a))),
            (Val::Qty(q), Val::Range(lo, hi)) if q.dim == Dim::Plain => {
                Ok(Val::Bool(q.v >= lo && q.v <= hi))
            }
            (Val::Qty(q), _) if q.dim == Dim::Plain => {
                Err(self.err_hint(line, "in looks in a list, a word, or a range", "like 3 in [1, 2, 3]"))
            }
            (l, r) => Err(self.err_hint(
                line,
                format!("in checks if a {} is inside a {}", val_name(&l), val_name(&r)),
                "like 3 in [1, 2, 3], or \"cup\" in \"cupcake\"",
            )),
        }
    }

    /// Indexing and slicing (2.1): xs[0], xs[1..3], word[2], r[1].
    fn eval_index(&mut self, base: &Expr, idx: &Expr, line: usize) -> Result<Val, Error> {
        let bv = self.eval_expr(base, line)?;
        // range index → slice
        if let Expr::Range(_, _) = idx {
            let iv = self.eval_expr(idx, line)?;
            let (lo, hi) = match iv {
                Val::Range(lo, hi) => (lo, hi),
                _ => return Err(self.err(line, "a slice is two numbers — like xs[1..3]")),
            };
            return match bv {
                Val::List(items) => {
                    let n = items.len() as i64;
                    let lo = clamp_idx(lo as i64, n);
                    let hi = clamp_idx(hi as i64, n).min(n - 1);
                    if lo > hi { return Ok(Val::List(Vec::new())); }
                    Ok(Val::List(items[lo as usize..=(hi as usize)].to_vec()))
                }
                Val::Str(s) => {
                    let chars: Vec<char> = s.chars().collect();
                    let n = chars.len() as i64;
                    let lo = clamp_idx(lo as i64, n);
                    let hi = clamp_idx(hi as i64, n).min(n - 1);
                    if lo > hi { return Ok(Val::Str(String::new())); }
                    Ok(Val::Str(chars[lo as usize..=(hi as usize)].iter().collect()))
                }
                other => Err(self.err_hint(
                    line,
                    format!("you slice lists and words, not a {}", val_name(&other)),
                    "like tail = xs[1..3]",
                )),
            };
        }
        let iv = self.eval_expr(idx, line)?;
        let i = match iv {
            Val::Qty(q) if q.dim == Dim::Plain => q.v.round() as i64,
            _ => return Err(self.err_hint(line, "an index is a plain number — like xs[0]", "lists count from 0; -1 means the last one")),
        };
        let shown = i; // the number the user wrote, for the error message
        match bv {
            Val::List(items) => {
                let n = items.len() as i64;
                let i = if i < 0 { i + n } else { i };
                if i < 0 || i >= n {
                    return Err(self.err_hint(
                        line,
                        format!("xs[{}] — that list has {} thing{}", shown, n, if n == 1 { "" } else { "s" }),
                        "lists count from 0; -1 picks the last one",
                    ));
                }
                Ok(items[i as usize].clone())
            }
            Val::Tuple(qs) => {
                let n = qs.len() as i64;
                let i = if i < 0 { i + n } else { i };
                if i < 0 || i >= n {
                    return Err(self.err_hint(line, format!("that position has {} numbers", n), "they count from 0"));
                }
                Ok(Val::Qty(qs[i as usize]))
            }
            Val::Str(s) => {
                let chars: Vec<char> = s.chars().collect();
                let n = chars.len() as i64;
                let i = if i < 0 { i + n } else { i };
                if i < 0 || i >= n {
                    return Err(self.err_hint(line, format!("that word has {} letters", n), "letters count from 0"));
                }
                Ok(Val::Str(chars[i as usize].to_string()))
            }
            Val::Range(lo, hi) => {
                let n = (hi - lo + 1.0).round();
                if i < 0 || i as f64 >= n {
                    return Err(self.err_hint(line, "that range isn't that long", "ranges count from their first number"));
                }
                Ok(Val::Qty(Qty::plain(lo + i as f64)))
            }
            Val::Shape(_) => Err(self.err_hint(line, "you can index lists, not objects", "make a list first: xs = [1, 2, 3]")),
            other => Err(self.err_hint(line, format!("a {} can't be indexed", val_name(&other)), "lists, positions, words and ranges can")),
        }
    }

    /// Truthiness for if/while/assert and logic (2.1).
    pub(super) fn truthy(&mut self, v: &Val, line: usize) -> Result<bool, Error> {
        match v {
            Val::Bool(b) => Ok(*b),
            Val::Qty(q) => Ok(q.v != 0.0),
            Val::Word(w) => match w.as_str() {
                "on" | "yes" => Ok(true),
                "off" | "no" | "none" => Ok(false),
                _ => Err(self.err_hint(line, format!("'{}' isn't true or false", w), "try a comparison like x > 5cm, or the words on / off")),
            },
            other => Err(self.err_hint(
                line,
                format!("if wants something true or false, not a {}", val_name(&other)),
                "try a comparison like x > 5cm, or a logic word like and / or / not",
            )),
        }
    }

    fn eval_hollow(&mut self, sv: &mut ShapeVal, args: &[Arg], line: usize) -> Result<ShapeVal, Error> {
        let mut wall = 2.0f64;
        // OTD4 — silent-wrongness fix: hollow() still defaults to Open::Top
        // (for backward compat with OTD3 files), but now we EMIT A HINT
        // every time `open:` is not explicit, so the user/AI is never
        // surprised. To get a sealed shell, write `open: none`; to breach
        // the bottom, `open: bottom`.
        let mut open = Open::Top;
        let mut open_was_set = false;
        let mut has_shape = false;
        let mut shape_arg: Option<Val> = None;
        for a in args {
            match a.name.as_deref() {
                Some("wall") => {
                    match self.eval_expr(&a.val, line)? {
                        Val::Qty(q) => {
                            let (w, hint) = check_positive(self.to_len(&q), "wall");
                            if let Some(h) = hint {
                                self.err_hint(line, "wall can't be zero or negative", h);
                            }
                            wall = w;
                        }
                        _ => return Err(self.err(line, "wall needs a thickness, like wall: 3mm")),
                    }
                }
                Some("open") => {
                    match self.eval_expr(&a.val, line)? {
                        Val::Word(w) => {
                            open = match w.as_str() {
                                "top" => Open::Top,
                                "bottom" => Open::Bottom,
                                "none" => Open::None,
                                _ => return Err(self.err_hint(line, "open wants top, bottom, or none", "try open: top")),
                            };
                            open_was_set = true;
                        }
                        _ => return Err(self.err(line, "open wants a word: top, bottom, or none")),
                    }
                }
                None | Some(_) => {
                    // positional: the shape (hollow(cube 10cm, wall: 5mm))
                    if !has_shape {
                        shape_arg = Some(self.eval_expr(&a.val, line)?);
                        has_shape = true;
                    }
                }
            }
        }
        let target: &mut ShapeVal = if let Some(Val::Shape(s)) = shape_arg {
            sv.meshes = s.meshes.clone();
            sv.kinds = s.kinds.clone();
            sv
        } else {
            sv
        };
        let mut out = ShapeVal { mat: target.mat, color: target.color, ..Default::default() };
        let mut approximated = false;
        for (m, k) in target.meshes.iter().zip(target.kinds.iter()) {
            let r: HollowResult = hollow(m, k, wall, open);
            if r.approximated {
                approximated = true;
            }
            out.meshes.push(r.mesh);
            out.kinds.push(Kind::Generic);
        }
        if out.meshes.is_empty() {
            out.meshes.push(Mesh::new());
            out.kinds.push(Kind::Generic);
        }
        if approximated {
            self.world.console.push(ConsoleLine {
                kind: LineKind::Info,
                text: "hollow used the general shell path (curved or fused shape) — walls are approximate".into(),
            });
        }
        // OTD4 — silent-wrongness fix: tell the user what default they got
        if !open_was_set {
            self.world.console.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "hollow() default: open top (write `open: none` for a sealed shell, or `open: bottom` to breach the base)"
                ),
            });
        }
        Ok(out)
    }

    // ---------- calls ----------

    fn eval_call(&mut self, name: &str, args: &[Arg], body: Option<&Expr>, line: usize) -> Result<Val, Error> {
        if kw::is_pattern(name) {
            return self.eval_pattern(name, args, body, line);
        }
        match name {
            "sphere" => self.shape_sphere(args, line),
            "cube" => self.shape_cube(args, line),
            "cylinder" => self.shape_cylinder(args, line),
            "cone" => self.shape_cone(args, line),
            "torus" => self.shape_torus(args, line),
            "pyramid" => self.shape_pyramid(args, line),
            "prism" => self.shape_prism(args, line),
            "capsule" => self.shape_capsule(args, line),
            "wedge" => self.shape_wedge(args, line),
            "plane" => self.shape_plane(args, line),
            "tube" => self.shape_tube(args, line),
            "helix" => self.shape_helix(args, line),
            "thread" => self.shape_thread(args, line),
            "extrude" => self.shape_extrude(args, line),
            "revolve" => self.shape_revolve(args, line),
            "sweep" => self.shape_sweep(args, line),
            "loft" => self.shape_loft(args, line),
            "text" => self.shape_text(args, line),
            "terrain" => self.shape_terrain(args, line),
            "metaball" => self.shape_metaball(args, line),
            "import" => self.shape_import(args, line),
            "blend" => self.shape_blend(args, line),
            "rope" => self.shape_rope(args, line),
            "hollow" => {
                let mut sv = ShapeVal::default();
                let out = self.eval_hollow(&mut sv, args, line)?;
                Ok(Val::Shape(out))
            }
            "group" => {
                // OTD4 — silent-wrongness fix: group() used to silently double-count
                // mass because the source parts stayed in entries AND their meshes
                // were re-added into the group. Now we re-parent: when an arg is
                // an Ident referring to an existing entry, that entry is marked
                // hidden (so it doesn't become a separate part) and the group
                // carries its meshes forward as the single combined part.
                let mut sv = ShapeVal::default();
                let mut absorbed_names: Vec<String> = Vec::new();
                for a in args {
                    // detect Ident reference to an existing entry — re-parent it
                    if let Expr::Ident(n) = &a.val {
                        if self.entries.iter().any(|e| e.name == *n) {
                            absorbed_names.push(n.clone());
                        }
                    }
                    if let Val::Shape(s) = self.eval_expr(&a.val, line)? {
                        sv.meshes.extend(s.meshes);
                        sv.kinds.extend(s.kinds);
                        if sv.mat.is_none() {
                            sv.mat = s.mat;
                            sv.color = s.color;
                        }
                    }
                }
                if !absorbed_names.is_empty() {
                    for n in &absorbed_names {
                        for e in self.entries.iter_mut() {
                            if e.name == *n {
                                e.hidden = true;
                            }
                        }
                    }
                    self.world.console.push(ConsoleLine {
                        kind: LineKind::Info,
                        text: format!(
                            "group() re-parented {} — they are now hidden as separate parts (the group is the single combined part). Old behavior double-counted their mass.",
                            absorbed_names.join(", ")
                        ),
                    });
                }
                Ok(Val::Shape(sv))
            }
            "add" | "subtract" | "intersect" => {
                // word form of the operators
                if args.len() != 2 {
                    return Err(self.err_hint(line, format!("{} takes two objects", name), "like subtract(cup, cube 2cm)"));
                }
                let a = self.eval_expr(&args[0].val, line)?;
                let b = self.eval_expr(&args[1].val, line)?;
                let op = if name == "add" { BinOp::Add } else if name == "subtract" { BinOp::Sub } else { BinOp::And };
                self.combine(op, a, b, line)
            }
            _ => {
                // math functions or template call
                if kw::is_func(name) {
                    return self.eval_func(name, args, line);
                }
                self.eval_template(name, args, line)
            }
        }
    }

    fn combine(&mut self, op: BinOp, a: Val, b: Val, line: usize) -> Result<Val, Error> {
        match (a, b) {
            (Val::Shape(mut a), Val::Shape(b)) => {
                let merged = match op {
                    BinOp::Add => {
                        let mut am = a.merged_mesh();
                        for m in &b.meshes {
                            am = mesh_union(&am, m);
                        }
                        am
                    }
                    BinOp::Sub => {
                        let am = a.merged_mesh();
                        let bm = b.merged_mesh();
                        mesh_subtract(&am, &bm)
                    }
                    BinOp::And => {
                        let am = a.merged_mesh();
                        let bm = b.merged_mesh();
                        mesh_intersect(&am, &bm)
                    }
                    _ => a.merged_mesh(),
                };
                a.meshes = vec![merged];
                a.kinds = vec![Kind::Generic];
                Ok(Val::Shape(a))
            }
            _ => Err(self.err_hint(line, format!("{} works on two objects", op_word(op)), "shapes go on both sides")),
        }
    }

    fn eval_func(&mut self, name: &str, args: &[Arg], line: usize) -> Result<Val, Error> {
        // evaluate every argument to a Val first (lists are values too, 2.1)
        let mut vals = Vec::new();
        for a in args {
            vals.push(self.eval_expr(&a.val, line)?);
        }
        // list-walking functions (2.1): count / sum / avg
        match name {
            "count" => {
                let n = match vals.first() {
                    Some(Val::List(vs)) => vs.len() as f64,
                    Some(Val::Range(lo, hi)) => (hi - lo + 1.0).max(0.0),
                    Some(Val::Tuple(qs)) => qs.len() as f64,
                    Some(Val::Str(s)) => s.chars().count() as f64,
                    Some(Val::Qty(q)) => q.v.max(0.0).round(),
                    Some(_) => return Err(self.err_hint(line, "count wants a list — like count(xs)", "lists look like [1, 2, 3]")),
                    None => return Err(self.err_hint(line, "count needs a list — like count(xs)", "lists look like [1, 2, 3]")),
                };
                return Ok(Val::Qty(Qty::plain(n)));
            }
            "sum" => {
                let mut acc = Qty::plain(0.0);
                let mut any = false;
                for v in &vals {
                    match v {
                        Val::List(vs) => {
                            for it in vs {
                                if let Val::Qty(q) = it {
                                    let (a, b) = self.promote_pair(acc, *q);
                                    acc = Qty { v: a.v + b.v, dim: a.dim.combine_add(b.dim).map_err(|e| self.plain_err(line, e))? };
                                    any = true;
                                }
                            }
                        }
                        Val::Range(lo, hi) => {
                            let n = (hi - lo + 1.0).max(0.0);
                            acc = Qty::plain(acc.v + (lo + hi) * n / 2.0);
                            any = true;
                        }
                        Val::Qty(q) => {
                            let (a, b) = self.promote_pair(acc, *q);
                            acc = Qty { v: a.v + b.v, dim: a.dim.combine_add(b.dim).map_err(|e| self.plain_err(line, e))? };
                            any = true;
                        }
                        _ => return Err(self.err_hint(line, "sum adds numbers — like sum([1, 2, 3])", "give it a list or plain numbers")),
                    }
                }
                let _ = any;
                return Ok(Val::Qty(acc));
            }
            "avg" => {
                let total = self.eval_func("sum", args, line)?;
                let n = self.eval_func("count", args, line)?;
                match (total, n) {
                    (Val::Qty(t), Val::Qty(c)) => {
                        if c.v == 0.0 {
                            return Err(self.err_hint(line, "avg of an empty list", "there is nothing in it to average"));
                        }
                        return Ok(Val::Qty(Qty { v: t.v / c.v, dim: t.dim }));
                    }
                    _ => unreachable!(),
                }
            }
            // OTD6 #1: in-script self-describing functions
            "help" => {
                self.world.console.push(ConsoleLine {
                    kind: LineKind::Info,
                    text: "OTD help — try: --list-functions, --list-materials, --list-keywords, --list-shapes, --list-simulate, --list-all from the CLI. In-script: functions(), materials(), keywords(), shapes(), sims()".into(),
                });
                return Ok(Val::Qty(Qty::plain(0.0)));
            }
            "functions" => {
                let fns = crate::lang::keywords::FUNCS;
                self.world.console.push(ConsoleLine { kind: LineKind::Info, text: format!("callable functions ({}): {}", fns.len(), fns.join("  ")) });
                return Ok(Val::Qty(Qty::plain(fns.len() as f64)));
            }
            "materials" => {
                let names = crate::world::materials::NAMES;
                self.world.console.push(ConsoleLine { kind: LineKind::Info, text: format!("materials ({}): {}", names.len(), names.join("  ")) });
                return Ok(Val::Qty(Qty::plain(names.len() as f64)));
            }
            "keywords" => {
                let kw = crate::lang::keywords::KEYWORDS;
                self.world.console.push(ConsoleLine { kind: LineKind::Info, text: format!("keywords ({}): {}", kw.len(), kw.join("  ")) });
                return Ok(Val::Qty(Qty::plain(kw.len() as f64)));
            }
            "shapes" => {
                let prims = crate::lang::keywords::PRIMITIVES;
                let builders = crate::lang::keywords::BUILDERS;
                self.world.console.push(ConsoleLine { kind: LineKind::Info, text: format!("primitives ({}): {}", prims.len(), prims.join("  ")) });
                self.world.console.push(ConsoleLine { kind: LineKind::Info, text: format!("builders ({}): {}", builders.len(), builders.join("  ")) });
                return Ok(Val::Qty(Qty::plain((prims.len() + builders.len()) as f64)));
            }
            "sims" => {
                let sims = "drop float collapse splash settle solidity gas mix energy heat magnet sound light time learn stats orbit atom decay particles motor circuit";
                self.world.console.push(ConsoleLine { kind: LineKind::Info, text: format!("simulate domains: {}", sims) });
                return Ok(Val::Qty(Qty::plain(21.0)));
            }
            _ => {}
        }
        let mut nums = Vec::new();
        for v in vals {
            match v {
                Val::Qty(q) => nums.push(q),
                _ => return Err(self.err(line, format!("{} wants plain numbers", name))),
            }
        }
        if nums.is_empty() {
            return Err(self.err_hint(line, format!("{} needs at least one number", name), "like sqrt(2)"));
        }
        let f = |q: &Qty| -> f64 {
            let x = match q.dim {
                Dim::Angle => q.v, // degrees
                _ => q.v,
            };
            match name {
                "cos" => x.to_radians().cos(),
                "sin" => x.to_radians().sin(),
                "tan" => x.to_radians().tan(),
                "sqrt" => if x >= 0.0 { x.sqrt() } else { f64::NAN },
                "abs" => x.abs(),
                "round" => x.round(),
                "floor" => x.floor(),
                "ceil" => x.ceil(),
                "sign" => if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
                "log" => if x > 0.0 { x.log10() } else { f64::NAN },
                "ln" => if x > 0.0 { x.ln() } else { f64::NAN },
                "exp" => x.exp(),
                "atan" => x.atan().to_degrees(),
                "asin" => if x >= -1.0 && x <= 1.0 { x.asin().to_degrees() } else { f64::NAN },
                "acos" => if x >= -1.0 && x <= 1.0 { x.acos().to_degrees() } else { f64::NAN },
                _ => x,
            }
        };
        let v = match name {
            "min" => nums.iter().map(|q| q.v).fold(f64::MAX, f64::min),
            "max" => nums.iter().map(|q| q.v).fold(f64::MIN, f64::max),
            "pow" => {
                if nums.len() != 2 {
                    return Err(self.err_hint(line, "pow takes two numbers — like pow(2, 10)", "or use ^: 2 ^ 10"));
                }
                if nums[0].dim != Dim::Plain || nums[1].dim != Dim::Plain {
                    return Err(self.err_hint(line, "pow works on plain numbers", "squaring a length would make an area"));
                }
                nums[0].v.powf(nums[1].v)
            }
            "hypot" => {
                if nums.len() != 2 {
                    return Err(self.err_hint(line, "hypot takes two numbers — like hypot(3, 4)", "it is the long side of a right triangle"));
                }
                nums[0].v.hypot(nums[1].v)
            }
            "atan2" => {
                if nums.len() != 2 {
                    return Err(self.err_hint(line, "atan2 takes two numbers — like atan2(1, 1)", "it gives the angle between them, in degrees"));
                }
                nums[0].v.atan2(nums[1].v).to_degrees()
            }
            "lerp" => {
                if nums.len() != 3 {
                    return Err(self.err_hint(line, "lerp blends between two numbers — like lerp(0cm, 10cm, 0.5)", "the last number says how far between (0 to 1)"));
                }
                let (a, b) = self.promote_pair(nums[0], nums[1]);
                let t = nums[2].v;
                return Ok(Val::Qty(Qty { v: a.v + (b.v - a.v) * t, dim: a.dim.combine_add(b.dim).map_err(|e| self.plain_err(line, e))? }));
            }
            "clamp" => {
                if nums.len() != 3 {
                    return Err(self.err_hint(line, "clamp keeps a number between two walls — like clamp(x, 0, 10)", "clamp(the number, the floor, the ceiling)"));
                }
                let (a, lo) = self.promote_pair(nums[0], nums[1]);
                let (a, hi) = self.promote_pair(a, nums[2]);
                return Ok(Val::Qty(Qty { v: a.v.max(lo.v).min(hi.v), dim: a.dim }));
            }
            // OTD6 #3: wire up electrodynamics functions as callable from scripts
            "ohm_v" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "ohm_v(i, r) — Ohm's law V=IR", "like ohm_v(2, 5) = 10")); }
                return Ok(Val::Qty(Qty::plain(nums[0].v * nums[1].v)));
            }
            "ohm_i" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "ohm_i(v, r) — Ohm's law I=V/R", "like ohm_i(10, 5) = 2")); }
                if nums[1].v.abs() < 1e-12 { return Err(self.err_hint(line, "ohm_i: resistance cannot be zero", "division by zero")); }
                return Ok(Val::Qty(Qty::plain(nums[0].v / nums[1].v)));
            }
            "ohm_r" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "ohm_r(v, i) — Ohm's law R=V/I", "like ohm_r(10, 2) = 5")); }
                if nums[1].v.abs() < 1e-12 { return Err(self.err_hint(line, "ohm_r: current cannot be zero", "division by zero")); }
                return Ok(Val::Qty(Qty::plain(nums[0].v / nums[1].v)));
            }
            "power_vi" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "power_vi(v, i) — electrical power P=VI", "like power_vi(12, 2) = 24")); }
                return Ok(Val::Qty(Qty::plain(nums[0].v * nums[1].v)));
            }
            "power_ir" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "power_ir(i, r) — power loss P=I²R", "like power_ir(2, 5) = 20")); }
                return Ok(Val::Qty(Qty::plain(nums[0].v * nums[0].v * nums[1].v)));
            }
            "cap_energy" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "cap_energy(c, v) — capacitor energy U=½CV²", "like cap_energy(1e-6, 12)")); }
                return Ok(Val::Qty(Qty::plain(0.5 * nums[0].v * nums[1].v * nums[1].v)));
            }
            "ind_energy" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "ind_energy(l, i) — inductor energy U=½LI²", "like ind_energy(1e-6, 2)")); }
                return Ok(Val::Qty(Qty::plain(0.5 * nums[0].v * nums[1].v * nums[1].v)));
            }
            "rc_tau" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "rc_tau(r, c) — RC time constant τ=RC", "like rc_tau(1000, 1e-6) = 0.001")); }
                return Ok(Val::Qty(Qty::plain(nums[0].v * nums[1].v)));
            }
            "lc_omega" => {
                if nums.len() != 2 { return Err(self.err_hint(line, "lc_omega(l, c) — LC resonance ω=1/√(LC)", "like lc_omega(1e-6, 1e-6)")); }
                if nums[0].v <= 0.0 || nums[1].v <= 0.0 { return Err(self.err_hint(line, "lc_omega: L and C must be positive", "")); }
                return Ok(Val::Qty(Qty::plain(1.0 / (nums[0].v * nums[1].v).sqrt())));
            }
            _ => f(&nums[0]),
        };
        // inverse trig answers in degrees
        let dim = if matches!(name, "atan" | "asin" | "acos" | "atan2") { Dim::Angle } else { Dim::Plain };
        Ok(Val::Qty(Qty { v, dim }))
    }

    fn eval_template(&mut self, name: &str, args: &[Arg], line: usize) -> Result<Val, Error> {
        let t = match self.templates.get(name).cloned() {
            Some(t) => t,
            None => {
                return Err(self.err_hint(
                    line,
                    format!("I don't know a shape or template called '{}'", name),
                    "define it first: define wheel = sphere 2cm",
                ))
            }
        };
        if self.depth > 24 {
            return Err(self.err_hint(line, "templates are calling each other in a loop", "a template can't use itself inside itself"));
        }
        // bind params
        let saved_env: Vec<(String, Val)> = t.params.iter().filter_map(|p| self.env.get(p).map(|v| (p.clone(), v.clone()))).collect();
        let mut bound = 0usize;
        for a in args {
            match a.name.as_deref() {
                Some(n) => {
                    if !t.params.contains(&n.to_string()) {
                        self.err_hint(line, format!("template '{}' has no setting '{}'", name, n), format!("settings: {}", t.params.join(", ")));
                        continue;
                    }
                    let v = self.eval_expr(&a.val, line)?;
                    self.env_insert(n.to_string(), v);
                    bound += 1;
                }
                None => {
                    if bound < t.params.len() {
                        let v = self.eval_expr(&a.val, line)?;
                        self.env_insert(t.params[bound].clone(), v);
                        bound += 1;
                    }
                }
            }
        }
        self.depth += 1;
        let result = if let Some(body) = &t.body {
            self.eval_expr(body, t.line)
        } else {
            // block template (2.1): run statements in a private scope; the last
            // value produced (expression or assignment) is the template's value
            let prev_tpl_mode = self.tpl_mode;
            let prev_val = self.tpl_val.take();
            let prev_journal = self.journal.take();
            self.journal = Some(Vec::new());
            self.tpl_mode = true;
            for s in &t.stmts {
                self.vm_fresh();
                self.exec(s.clone());
            }
            let val = self.tpl_val.take();
            self.tpl_mode = prev_tpl_mode;
            self.tpl_val = prev_val;
            // undo every env write the body made (params + locals)
            if let Some(j) = self.journal.take() {
                for (k, old) in j.into_iter().rev() {
                    match old {
                        Some(v) => { self.env.insert(k, v); }
                        None => { self.env.remove(&k); }
                    }
                }
            }
            self.journal = prev_journal;
            match val {
                Some(v) => Ok(v),
                None => Err(self.err_hint(
                    line,
                    format!("define {} … end should end with the shape it makes", name),
                    "the last line before end is the value — put it there alone",
                )),
            }
        };
        self.depth -= 1;
        // restore shadowed params
        for (k, v) in saved_env {
            self.env.insert(k, v);
        }
        result
    }

    // ---------- patterns ----------

    fn eval_pattern(&mut self, name: &str, args: &[Arg], body: Option<&Expr>, line: usize) -> Result<Val, Error> {
        let body = match body {
            Some(b) => b,
            None => return Err(self.err_hint(line, format!("{} needs a shape after it", name), "like repeat(n: 4) cube 1cm")),
        };
        // named settings
        let mut n: u32 = if name == "ring" { 8 } else if name == "scatter" { 20 } else { 4 };
        let mut nx: u32 = 3;
        let mut nz: u32 = 3;
        let mut step = V3::ZERO;
        let mut spacing = V3::ZERO;
        let mut radius = 0f64;
        let mut from = 0f64;
        let mut to = 360f64;
        let mut axis = 'y';
        let mut seed = 7u64;
        let mut has_step = false;
        let mut has_spacing = false;
        let mut has_radius = false;
        for a in args {
            match a.name.as_deref() {
                Some("n") => match self.eval_expr(&a.val, line)? {
                    Val::Qty(q) => n = q.expect_plain("n").max(1.0).min(5000.0) as u32,
                    _ => return Err(self.err(line, "n wants a count, like n: 8")),
                },
                Some("nx") => match self.eval_expr(&a.val, line)? {
                    Val::Qty(q) => nx = q.expect_plain("nx").max(1.0).min(200.0) as u32,
                    _ => return Err(self.err(line, "nx wants a count")),
                },
                Some("nz") => match self.eval_expr(&a.val, line)? {
                    Val::Qty(q) => nz = q.expect_plain("nz").max(1.0).min(200.0) as u32,
                    _ => return Err(self.err(line, "nz wants a count")),
                },
                Some("step") => {
                    step = self.tuple3(&a.val, line)?;
                    has_step = true;
                }
                Some("spacing") => {
                    spacing = self.tuple2xz(&a.val, line)?;
                    has_spacing = true;
                }
                Some("radius") => match self.eval_expr(&a.val, line)? {
                    Val::Qty(q) => {
                        let (r, hint) = check_positive(self.to_len(&q), "radius");
                        if let Some(h) = hint {
                            return Err(self.err_hint(line, "radius cannot be negative", h));
                        }
                        radius = r;
                        has_radius = true;
                    }
                    _ => return Err(self.err(line, "radius wants a length, like radius: 5cm")),
                },
                Some("from") => match self.eval_expr(&a.val, line)? {
                    Val::Qty(q) => from = q.expect_angle("from"),
                    _ => return Err(self.err(line, "from wants an angle, like from: 0deg")),
                },
                Some("to") => match self.eval_expr(&a.val, line)? {
                    Val::Qty(q) => to = q.expect_angle("to"),
                    _ => return Err(self.err(line, "to wants an angle, like to: 180deg")),
                },
                Some("axis") => match self.eval_expr(&a.val, line)? {
                    Val::Word(w) => axis = w.chars().next().unwrap_or('y'),
                    _ => return Err(self.err_hint(line, "axis wants x, y, or z", "like axis: z")),
                },
                Some("seed") => match self.eval_expr(&a.val, line)? {
                    Val::Qty(q) => seed = q.expect_plain("seed").max(0.0) as u64,
                    _ => return Err(self.err(line, "seed wants a plain number, like seed: 7")),
                },
                _ => {}
            }
        }
        if name == "ring" && !has_radius {
            // auto radius from the body bbox after one probe
            if let Ok(Val::Shape(sv)) = self.eval_expr(body, line) {
                let bb = sv.bbox();
                radius = bb.size().x().max(bb.size().z()).max(1.0);
            }
        }
        if name == "ring" && !has_radius && radius <= 0.0 {
            return Err(self.err_hint(line, "ring needs a radius", "like ring(n: 8, radius: 5cm) sphere 1cm"));
        }
        if name == "grid" && !has_spacing {
            if let Ok(Val::Shape(sv)) = self.eval_expr(body, line) {
                let bb = sv.bbox();
                spacing = V3::new(bb.size().x() * 1.1, 0.0, bb.size().z() * 1.1);
            }
        }
        let _ = (has_step, has_spacing);

        let total = if name == "grid" { nx * nz } else { n };
        let mut out = ShapeVal::default();
        let mut rng = crate::rng::Rng::new(seed);
        for idx in 0..total {
            // bind magic vars
            let a_deg = from + (to - from) * (idx as f64) / (n as f64);
            self.magic.retain(|(k, _)| k != "i" && k != "j" && k != "a");
            match name {
                "repeat" => self.magic.push(("i".into(), Qty::plain(idx as f64))),
                "grid" => {
                    self.magic.push(("i".into(), Qty::plain((idx % nx) as f64)));
                    self.magic.push(("j".into(), Qty::plain((idx / nx) as f64)));
                }
                "ring" | "scatter" => {
                    self.magic.push(("i".into(), Qty::plain(idx as f64)));
                    self.magic.push(("a".into(), Qty::deg(a_deg)));
                }
                _ => {}
            }
            if let Ok(Val::Shape(mut sv)) = self.eval_expr(body, line) {
                // per-pattern transforms
                let offset = match name {
                    "repeat" => step.mul(idx as f64),
                    "grid" => {
                        let (ix, iz) = ((idx % nx) as f64, (idx / nx) as f64);
                        V3::new(spacing.x() * ix, 0.0, spacing.z() * iz)
                    }
                    "ring" => {
                        let (ar, ax) = (a_deg.to_radians(), axis);
                        match ax {
                            'x' => V3::new(0.0, radius * ar.cos(), radius * ar.sin()),
                            'z' => V3::new(radius * ar.cos(), radius * ar.sin(), 0.0),
                            _ => V3::new(radius * ar.cos(), 0.0, -radius * ar.sin()),
                        }
                    }
                    "scatter" => {
                        let ang = rng.next_f64() * 2.0 * std::f64::consts::PI;
                        let dist = radius * rng.next_f64().sqrt();
                        V3::new(dist * ang.cos(), 0.0, dist * ang.sin())
                    }
                    _ => V3::ZERO,
                };
                if offset.len() > 1e-9 {
                    sv.translate(offset);
                }
                if out.mat.is_none() {
                    out.mat = sv.mat;
                    out.color = sv.color;
                }
                out.meshes.extend(sv.meshes);
                out.kinds.extend(sv.kinds);
            }
        }
        self.magic.clear();
        if out.meshes.is_empty() {
            return Err(self.err_hint(line, format!("{} made nothing", name), "the shape after it may have failed"));
        }
        Ok(Val::Shape(out))
    }

    fn tuple3(&mut self, e: &Expr, line: usize) -> Result<V3, Error> {
        match self.eval_expr(e, line)? {
            Val::Tuple(qs) if qs.len() == 3 => Ok(V3::new(
                self.to_len(&qs[0]),
                self.to_len(&qs[1]),
                self.to_len(&qs[2]),
            )),
            _ => Err(self.err_hint(line, "this wants a position (x, y, z)", "like (5cm, 0, 0)")),
        }
    }

    fn tuple2xz(&mut self, e: &Expr, line: usize) -> Result<V3, Error> {
        match self.eval_expr(e, line)? {
            Val::Tuple(qs) if qs.len() == 2 => Ok(V3::new(self.to_len(&qs[0]), 0.0, self.to_len(&qs[1]))),
            Val::Tuple(qs) if qs.len() == 3 => Ok(V3::new(self.to_len(&qs[0]), self.to_len(&qs[1]), self.to_len(&qs[2]))),
            _ => Err(self.err_hint(line, "spacing wants (x, z) or (x, y, z)", "like spacing: (10cm, 10cm)")),
        }
    }

    // ---------- modifiers ----------

    fn apply_mod(&mut self, sv: &mut ShapeVal, m: &Mod, line: usize) {
        match m {
            Mod::At(items) => {
                if items.len() < 2 {
                    self.err_hint(line, "at wants a position", "like at (4cm, 5cm, 0)");
                    return;
                }
                let mut pos = [0.0f64; 3];
                for (k, item) in items.iter().take(3).enumerate() {
                    if let Ok(Val::Qty(q)) = self.eval_expr(item, line) {
                        pos[k] = match q.dim {
                            Dim::Angle => {
                                self.err_hint(line, "at wants lengths, not angles", format!("did you mean {}mm?", q.v));
                                q.v
                            }
                            _ => self.to_len(&q),
                        };
                    }
                }
                if items.len() == 2 {
                    // (x, z) on the ground
                    pos = [pos[0], 0.0, pos[1]];
                }
                let bb = sv.bbox();
                let c = bb.center();
                sv.translate(V3::new(pos[0] - c.x(), pos[1] - bb.min.y(), pos[2] - c.z()));
            }
            Mod::Rotate(v) => {
                // OTD4 — silent-wrongness fix: the default pivot is the
                // bounding-box CENTER. This is documented in the hint the
                // first time rotate is used without `pivot:`. Use
                // `rotate (...) pivot (0, 0, 0)` to pivot around the origin,
                // or `pivot: center` to be explicit.
                let bb = sv.bbox();
                let c = bb.center();
                self.world.console.push(ConsoleLine {
                    kind: LineKind::Info,
                    text: format!(
                        "rotate: pivot = bounding-box center ({:.1}, {:.1}, {:.1}) mm — write `pivot (0,0,0)` to pivot around the origin, or `pivot: center` to be explicit",
                        c.x(), c.y(), c.z()
                    ),
                });
                let m4 = match self.eval_expr(v, line) {
                    Ok(Val::Tuple(qs)) => {
                        let (mut rx, mut ry, mut rz) = (0.0, 0.0, 0.0);
                        match qs.len() {
                            3 => {
                                rx = qs[0].expect_angle("x rotation");
                                ry = qs[1].expect_angle("y rotation");
                                rz = qs[2].expect_angle("z rotation");
                            }
                            _ => { self.err_hint(line, "rotate wants one angle or (x, y, z) angles", "like rotate (0, 45deg, 0)"); }
                        }
                        M4::translate(-c.x(), -c.y(), -c.z())
                            .mul(&M4::rot_z(rz))
                            .mul(&M4::rot_y(ry))
                            .mul(&M4::rot_x(rx))
                            .mul(&M4::translate(c.x(), c.y(), c.z()))
                    }
                    Ok(Val::Qty(q)) => {
                        let a = q.expect_angle("rotation");
                        // bare angle spins about the vertical axis
                        M4::translate(-c.x(), -c.y(), -c.z())
                            .mul(&M4::rot_y(a))
                            .mul(&M4::translate(c.x(), c.y(), c.z()))
                    }
                    _ => {
                        self.err_hint(line, "rotate wants an angle or a tuple of angles", "like rotate 90deg or rotate (0, 45deg, 0)");
                        return;
                    }
                };
                for mesh in &mut sv.meshes {
                    mesh.transform(&m4);
                }
            }
            // OTD4 — `rotate (...) pivot (...)` — explicit pivot point
            Mod::RotateWithPivot { angles, pivot } => {
                let bb = sv.bbox();
                let center = bb.center();
                let pivot_pt = match pivot {
                    crate::lang::ast::PivotSpec::Origin => V3::ZERO,
                    crate::lang::ast::PivotSpec::Center => center,
                    crate::lang::ast::PivotSpec::Point(items) => {
                        let mut p = [0.0f64; 3];
                        for (k, item) in items.iter().take(3).enumerate() {
                            if let Ok(Val::Qty(q)) = self.eval_expr(item, line) {
                                p[k] = self.to_len(&q);
                            }
                        }
                        V3::new(p[0], p[1], p[2])
                    }
                };
                let m4 = match self.eval_expr(angles, line) {
                    Ok(Val::Tuple(qs)) => {
                        let (mut rx, mut ry, mut rz) = (0.0, 0.0, 0.0);
                        match qs.len() {
                            3 => {
                                rx = qs[0].expect_angle("x rotation");
                                ry = qs[1].expect_angle("y rotation");
                                rz = qs[2].expect_angle("z rotation");
                            }
                            _ => { self.err_hint(line, "rotate wants one angle or (x, y, z) angles", "like rotate (0, 45deg, 0)"); }
                        }
                        M4::translate(-pivot_pt.x(), -pivot_pt.y(), -pivot_pt.z())
                            .mul(&M4::rot_z(rz))
                            .mul(&M4::rot_y(ry))
                            .mul(&M4::rot_x(rx))
                            .mul(&M4::translate(pivot_pt.x(), pivot_pt.y(), pivot_pt.z()))
                    }
                    Ok(Val::Qty(q)) => {
                        let a = q.expect_angle("rotation");
                        M4::translate(-pivot_pt.x(), -pivot_pt.y(), -pivot_pt.z())
                            .mul(&M4::rot_y(a))
                            .mul(&M4::translate(pivot_pt.x(), pivot_pt.y(), pivot_pt.z()))
                    }
                    _ => {
                        self.err_hint(line, "rotate wants an angle or a tuple of angles", "like rotate 90deg or rotate (0, 45deg, 0) pivot (0, 0, 0)");
                        return;
                    }
                };
                self.world.console.push(ConsoleLine {
                    kind: LineKind::Info,
                    text: format!(
                        "rotate: pivot = ({:.1}, {:.1}, {:.1}) mm (explicit)",
                        pivot_pt.x(), pivot_pt.y(), pivot_pt.z()
                    ),
                });
                for mesh in &mut sv.meshes {
                    mesh.transform(&m4);
                }
            }
            Mod::Scale(v) => {
                let bb = sv.bbox();
                let c = bb.center();
                let (sx, sy, sz) = match self.eval_expr(v, line) {
                    Ok(Val::Qty(q)) => {
                        let f = q.expect_plain("scale");
                        (f, f, f)
                    }
                    Ok(Val::Tuple(qs)) if qs.len() == 3 => (
                        qs[0].expect_plain("scale x"),
                        qs[1].expect_plain("scale y"),
                        qs[2].expect_plain("scale z"),
                    ),
                    _ => {
                        self.err_hint(line, "scale wants a number or (x, y, z) factors", "like scale 2 or scale (2, 1, 1)");
                        return;
                    }
                };
                let m4 = M4::translate(-c.x(), -c.y(), -c.z())
                    .mul(&M4::scale(sx, sy, sz))
                    .mul(&M4::translate(c.x(), c.y(), c.z()));
                for mesh in &mut sv.meshes {
                    mesh.transform(&m4);
                }
            }
            Mod::Mirror(ax) => {
                let (sx, sy, sz) = match ax {
                    'x' => (-1.0, 1.0, 1.0),
                    'y' => (1.0, -1.0, 1.0),
                    _ => (1.0, 1.0, -1.0),
                };
                let m4 = M4::scale(sx, sy, sz);
                for mesh in &mut sv.meshes {
                    mesh.transform(&m4);
                }
                for mesh in &mut sv.meshes {
                    mesh.ensure_outward();
                }
            }
            Mod::Material(name) => {
                match materials::find(name) {
                    Some(mm) => sv.mat = Some(mm),
                    None => {
                        let hint = materials::suggest(name)
                            .map(|s| format!("did you mean {}?", s))
                            .unwrap_or_else(|| "materials are words like steel, oak, ceramic".into());
                        self.err_hint(line, format!("'{}' is not a material I know", name), hint);
                    }
                }
            }
            Mod::Color(name) => {
                match colors::resolve(name) {
                    Some(c) => sv.color = Some(c),
                    None => {
                        let hint = colors::suggest(name)
                            .map(|s| format!("did you mean {}?", s))
                            .unwrap_or_else(|| "colors are words like ivory, or #rrggbb".into());
                        self.err_hint(line, format!("'{}' is not a color I know", name), hint);
                    }
                }
            }
            Mod::Smooth { iterations, strength } => {
                // P0900 — the DEC tier: cotan-Laplacian Taubin relaxation
                let n = match self.eval_expr(iterations, line) {
                    Ok(Val::Qty(q)) => q.expect_plain("n").max(0.0).min(50.0) as u32,
                    _ => {
                        self.err_hint(line, "smooth wants a count", "like smooth(n: 3)");
                        return;
                    }
                };
                let strength = match self.eval_expr(strength, line) {
                    Ok(Val::Qty(q)) => q.expect_plain("strength").clamp(0.0, 1.0),
                    _ => 0.5,
                };
                if n == 0 {
                    return;
                }
                let mut out = Vec::with_capacity(sv.meshes.len());
                for mesh in &sv.meshes {
                    out.push(crate::geo::dec::smooth(mesh, n, strength));
                }
                sv.meshes = out;
                sv.kinds = vec![Kind::Generic; sv.meshes.len()];
            }
            Mod::Subdiv(levels) => {
                // P1000 — the topology tier: Loop subdivision
                let n = match self.eval_expr(levels, line) {
                    Ok(Val::Qty(q)) => q.expect_plain("n").max(0.0).min(4.0) as u32,
                    _ => {
                        self.err_hint(line, "subdiv wants a level count", "like subdiv(n: 2) — each level is 4× the triangles");
                        return;
                    }
                };
                if n == 0 {
                    return;
                }
                let mut out = Vec::with_capacity(sv.meshes.len());
                for mesh in &sv.meshes {
                    out.push(crate::geo::subdiv::loop_subdivide(mesh, n));
                }
                sv.meshes = out;
                sv.kinds = vec![Kind::Generic; sv.meshes.len()];
                self.world.console.push(ConsoleLine {
                    kind: LineKind::Info,
                    text: format!("subdiv(n: {}) — the surface is smoother now ({} triangles)", n,
                        sv.meshes.iter().map(|m| m.tris.len()).sum::<usize>()),
                });
            }
        }
    }
}

fn clamp_idx(mut i: i64, n: i64) -> i64 {
    if i < 0 { i += n; }
    if i < 0 { 0 } else if i > n - 1 { n - 1 } else { i }
}

fn check_positive(v: f64, what: &str) -> (f64, Option<String>) {
    if v < 0.0 {
        (v.abs(), Some(format!("did you mean {}mm?", v.abs())))
    } else if v == 0.0 {
        (1.0, Some(format!("{} can't be zero — using 1mm", what)))
    } else {
        (v, None)
    }
}

fn val_name(v: &Val) -> &'static str {
    match v {
        Val::Qty(_) => "number",
        Val::Str(_) => "text",
        Val::Tuple(_) => "position",
        Val::List(_) => "list",
        Val::Bool(_) => "true/false",
        Val::Range(..) => "range",
        Val::Shape(_) => "object",
        Val::Word(_) => "word",
    }
}

/// Short human rendering of a quantity for error messages.
fn fmt_qty(q: Qty) -> String {
    match q.dim {
        Dim::Plain => format_num(q.v),
        Dim::Length => format!("{}mm", format_num(q.v)),
        Dim::Angle => format!("{}deg", format_num(q.v)),
    }
}

fn format_num(v: f64) -> String {
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        let s = format!("{:.3}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

/// Human rendering of any value for `print "… {name}"` interpolation (2.1).
fn format_val(v: &Val) -> String {
    match v {
        Val::Qty(q) => fmt_qty(*q),
        Val::Str(s) => s.clone(),
        Val::Bool(b) => if *b { "true".into() } else { "false".into() },
        Val::Word(w) => w.clone(),
        Val::Tuple(qs) => {
            let parts: Vec<String> = qs.iter().map(|q| fmt_qty(*q)).collect();
            format!("({})", parts.join(", "))
        }
        Val::List(vs) => {
            let n = vs.len();
            format!("a list of {} thing{}", n, if n == 1 { "" } else { "s" })
        }
        Val::Range(lo, hi) => format!("{}..{}", format_num(*lo), format_num(*hi)),
        Val::Shape(_) => "a shape".into(),
    }
}

fn op_word(op: BinOp) -> &'static str {
    match op {
        BinOp::Add => "add",
        BinOp::Sub => "subtract",
        BinOp::And => "intersect",
        BinOp::Mul => "multiply",
        BinOp::Div => "divide",
        BinOp::Pow => "raise",
        BinOp::Mod => "take the remainder of",
        BinOp::Lt | BinOp::Le => "compare",
        BinOp::Gt | BinOp::Ge => "compare",
        BinOp::Eq | BinOp::Ne => "compare",
        BinOp::AndAlso => "and",
        BinOp::OrElse => "or",
        BinOp::In => "look for",
    }
}

// Dim::combine_add helper
impl Dim {
    fn combine_add(self, other: Dim) -> Result<Dim, String> {
        use Dim::*;
        match (self, other) {
            (Plain, x) | (x, Plain) => Ok(x),
            (a, b) if a == b => Ok(a),
            (Length, Angle) | (Angle, Length) => Err("you added a length to an angle — check the units".into()),
            (Length, Length) | (Angle, Angle) => Ok(self),
        }
    }
}
