//! Shape-call argument handling for the evaluator (primitives & builders).

use super::eval::{Ctx, ShapeVal, Val};
use crate::geo::mesh::Kind;
use crate::geo::prims;
use crate::lang::ast::{Arg, Expr};
use crate::world::eval::LineKind;
use crate::world::eval::ConsoleLine;

/// Evaluated call arguments: named map + positional queue.
pub struct ShapeArgs {
    pub named: Vec<(String, Val)>,
    pub pos: Vec<Val>,
    /// scene default unit × this = millimeters (bare numbers)
    pub unit_mm: f64,
}

impl<'a> Ctx<'a> {
    pub(super) fn collect_args(&mut self, args: &[Arg], line: usize) -> Result<ShapeArgs, crate::lang::errors::Error> {
        let mut named = Vec::new();
        let mut pos = Vec::new();
        for a in args {
            let v = match (a.name.as_deref(), &a.val) {
                // `material: rubber` / `color: #1e90ff` — value-library words, not
                // expressions. A variable of the same name still wins (templates).
                (Some("material") | Some("color"), Expr::Ident(w)) => {
                    if self.env.contains_key(w) {
                        self.eval_expr(&a.val, line)?
                    } else {
                        Val::Word(w.clone())
                    }
                }
                _ => self.eval_expr(&a.val, line)?,
            };
            match a.name.as_deref() {
                Some(n) => named.push((n.to_string(), v)),
                None => pos.push(v),
            }
        }
        Ok(ShapeArgs { named, pos, unit_mm: self.unit_mm })
    }
}

impl ShapeArgs {
    fn named(&self, aliases: &[&str]) -> Option<&Val> {
        self.named.iter().find(|(k, _)| aliases.contains(&k.as_str())).map(|(_, v)| v)
    }
    fn take_pos(&mut self) -> Option<Val> {
        if self.pos.is_empty() { None } else { Some(self.pos.remove(0)) }
    }
    /// length in mm (named alias, else next positional, else default).
    /// Bare numbers get the scene default unit; explicit units pass through.
    pub fn take_len(&mut self, aliases: &[&str], default: f64, _what: &str) -> f64 {
        let unit = self.unit_mm;
        let to_mm = |v: &Val| -> Option<f64> {
            if let Val::Qty(q) = v {
                return Some(match q.dim {
                    crate::units::Dim::Length => q.v,
                    _ => q.v * unit,
                });
            }
            None
        };
        if let Some(v) = self.named(aliases) {
            if let Some(mm) = to_mm(v) {
                return mm;
            }
        }
        if let Some(v) = self.take_pos() {
            if let Some(mm) = to_mm(&v) {
                return mm;
            }
        }
        default
    }
    pub fn take_plain(&mut self, aliases: &[&str], default: f64, what: &str) -> f64 {
        if let Some(v) = self.named(aliases) {
            if let Val::Qty(q) = v {
                return q.expect_plain(what);
            }
        }
        if let Some(v) = self.take_pos() {
            if let Val::Qty(q) = v {
                return q.expect_plain(what);
            }
        }
        default
    }
    pub fn take_word(&mut self, aliases: &[&str], default: &str) -> String {
        if let Some(Val::Word(w)) = self.named(aliases) {
            return w.clone();
        }
        if let Some(Val::Word(w)) = self.take_pos() {
            return w.clone();
        }
        default.to_string()
    }
    pub fn take_str(&mut self, aliases: &[&str], default: &str) -> String {
        if let Some(Val::Str(s)) = self.named(aliases) {
            return s.clone();
        }
        if let Some(Val::Str(s)) = self.take_pos() {
            return s.clone();
        }
        default.to_string()
    }
    /// list of 2-tuples (mm, mm)
    pub fn take_profile(&mut self, aliases: &[&str]) -> Option<Vec<(f64, f64)>> {
        let v = self.named(aliases).cloned().or_else(|| self.take_pos())?;
        profile_of_unit(&v, self.unit_mm)
    }
    /// list of 3-tuples (mm)
    pub fn take_path(&mut self, aliases: &[&str]) -> Option<Vec<crate::math3::V3>> {
        let v = self.named(aliases).cloned().or_else(|| self.take_pos())?;
        path_of_unit(&v, self.unit_mm)
    }
    /// list of lists of 2-tuples
    pub fn take_sections(&mut self, aliases: &[&str]) -> Option<Vec<Vec<(f64, f64)>>> {
        let v = self.named(aliases).cloned().or_else(|| self.take_pos())?;
        if let Val::List(items) = v {
            let mut secs = Vec::new();
            for it in items {
                if let Some(p) = profile_of_unit(&it, self.unit_mm) {
                    secs.push(p);
                }
            }
            if !secs.is_empty() {
                return Some(secs);
            }
        }
        None
    }


}

pub fn profile_of(v: &Val) -> Option<Vec<(f64, f64)>> {
    profile_of_unit(v, 1.0)
}

/// Profile points in mm; bare numbers × unit_mm (scene default unit).
pub fn profile_of_unit(v: &Val, unit_mm: f64) -> Option<Vec<(f64, f64)>> {
    let to_mm = |q: &crate::units::Qty| -> f64 {
        match q.dim {
            crate::units::Dim::Length => q.v,
            _ => q.v * unit_mm,
        }
    };
    if let Val::List(items) = v {
        let mut pts = Vec::new();
        for it in items {
            if let Val::Tuple(qs) = it {
                if qs.len() >= 2 {
                    pts.push((to_mm(&qs[0]), to_mm(&qs[1])));
                }
            }
        }
        if !pts.is_empty() {
            return Some(pts);
        }
    }
    None
}

pub fn path_of(v: &Val) -> Option<Vec<crate::math3::V3>> {
    path_of_unit(v, 1.0)
}

/// Path points in mm; bare numbers × unit_mm.
pub fn path_of_unit(v: &Val, unit_mm: f64) -> Option<Vec<crate::math3::V3>> {
    let to_mm = |q: &crate::units::Qty| -> f64 {
        match q.dim {
            crate::units::Dim::Length => q.v,
            _ => q.v * unit_mm,
        }
    };
    // a single tuple is a one-point path — `rope(from: (0, 30cm, 0), …)`
    if let Val::Tuple(qs) = v {
        match qs.len() {
            2 => return Some(vec![crate::math3::V3::new(to_mm(&qs[0]), 0.0, to_mm(&qs[1]))]),
            n if n >= 3 => {
                return Some(vec![crate::math3::V3::new(
                    to_mm(&qs[0]),
                    to_mm(&qs[1]),
                    to_mm(&qs[2]),
                )])
            }
            _ => {}
        }
    }
    if let Val::List(items) = v {
        let mut pts = Vec::new();
        for it in items {
            if let Val::Tuple(qs) = it {
                if qs.len() >= 3 {
                    pts.push(crate::math3::V3::new(
                        to_mm(&qs[0]),
                        to_mm(&qs[1]),
                        to_mm(&qs[2]),
                    ));
                }
            }
        }
        if !pts.is_empty() {
            return Some(pts);
        }
    }
    None
}

fn clamp_smooth(v: f64) -> u32 {
    v.max(8.0).min(192.0) as u32
}

impl<'a> Ctx<'a> {
    /// Finish a shape call: attach material:/color: given as call arguments
    /// (e.g. `sphere(2cm, material: wood)`).
    fn finish(&mut self, m: crate::geo::mesh::Mesh, k: Kind, a: &ShapeArgs, line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut sv = ShapeVal::from_mesh(m, k);
        if let Some(v) = a.named(&["material"]) {
            if let Val::Word(w) = v {
                match crate::world::materials::find(w) {
                    Some(mm) => sv.mat = Some(mm),
                    None => {
                        let hint = crate::world::materials::suggest(w)
                            .map(|s| format!("did you mean {}?", s))
                            .unwrap_or_else(|| "materials are words like steel, oak, ceramic".into());
                        self.err_hint(line, format!("'{}' is not a material I know", w), hint);
                    }
                }
            }
        }
        if let Some(v) = a.named(&["color"]) {
            if let Val::Word(w) = v {
                match crate::world::colors::resolve(w) {
                    Some(c) => sv.color = Some(c),
                    None => {
                        let hint = crate::world::colors::suggest(w)
                            .map(|s| format!("did you mean {}?", s))
                            .unwrap_or_else(|| "colors are words like ivory, or #rrggbb".into());
                        self.err_hint(line, format!("'{}' is not a color I know", w), hint);
                    }
                }
            }
        }
        Ok(Val::Shape(sv))
    }

    pub(super) fn shape_sphere(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let r = a.take_len(&["r", "radius"], 10.0, "sphere radius");
        let smooth = a.take_plain(&["smooth"], 48.0, "smooth");
        let (r, hint) = friendly_positive(r, "radius");
        if let Some(h) = hint {
            self.err_hint(line, "radius cannot be negative", h);
        }
        let (m, k) = prims::sphere(r, clamp_smooth(smooth));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_cube(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        // cube 10cm | cube(w, d, h)
        let (w, d, h) = if a.pos.len() >= 3 {
            // FIX(P0210-bug): positional args must pop in the documented
            // cube(w, d, h) order. Popping h first silently swapped w and h,
            // laying every positional 3-arg cube on its side.
            let w = a.take_len(&["w", "width"], 20.0, "width");
            let d = a.take_len(&["d", "depth"], 20.0, "depth");
            let h = a.take_len(&["h", "height"], 20.0, "height");
            (w, d, h)
        } else {
            let s = a.take_len(&["s", "size", "w", "width"], 20.0, "cube size");
            let mut w = s;
            let mut d = s;
            let mut h = s;
            if let Some(v) = a.named(&["d", "depth"]) {
                if let Val::Qty(q) = v { d = q.expect_len("depth"); }
            }
            if let Some(v) = a.named(&["h", "height"]) {
                if let Val::Qty(q) = v { h = q.expect_len("height"); }
            }
            if let Some(v) = a.named(&["w", "width"]) {
                if let Val::Qty(q) = v { w = q.expect_len("width"); }
            }
            (w, d, h)
        };
        let dims = [w, d, h];
        for (i, v) in dims.iter().enumerate() {
            if *v <= 0.0 {
                let names = ["width", "depth", "height"];
                self.err_hint(line, format!("{} cannot be zero or negative ({}mm)", names[i], v), format!("did you mean {}mm?", v.abs()));
            }
        }
        let (m, k) = prims::cube(w.abs().max(0.1), d.abs().max(0.1), h.abs().max(0.1));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_cylinder(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let mut r = a.take_len(&["r", "radius"], 10.0, "radius");
        let mut h = a.take_len(&["h", "height"], 20.0, "height");
        let mut top = r;
        let mut bottom = r;
        if let Some(v) = a.named(&["top"]) {
            if let Val::Qty(q) = v { top = q.expect_len("top"); }
        }
        if let Some(v) = a.named(&["bottom"]) {
            if let Val::Qty(q) = v { bottom = q.expect_len("bottom"); }
        }
        let smooth = a.take_plain(&["smooth"], 48.0, "smooth");
        if top <= 0.0 && bottom <= 0.0 {
            self.err_hint(line, "a cylinder needs at least one radius above zero", format!("did you mean {}mm?", r.abs()));
            r = r.abs().max(1.0);
            top = r;
            bottom = r;
        }
        if h <= 0.0 {
            self.err_hint(line, format!("height cannot be negative ({}mm)", h), format!("did you mean {}mm?", h.abs()));
            h = h.abs().max(1.0);
        }
        let (m, k) = prims::frustum(top.max(0.0), bottom.max(0.0), h, clamp_smooth(smooth));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_cone(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let mut r = a.take_len(&["r", "radius"], 10.0, "radius");
        let mut h = a.take_len(&["h", "height"], 20.0, "height");
        let smooth = a.take_plain(&["smooth"], 48.0, "smooth");
        if r <= 0.0 {
            self.err_hint(line, "radius cannot be negative", format!("did you mean {}mm?", r.abs()));
            r = r.abs().max(1.0);
        }
        if h <= 0.0 {
            self.err_hint(line, format!("height cannot be negative ({}mm)", h), format!("did you mean {}mm?", h.abs()));
            h = h.abs().max(1.0);
        }
        let (m, k) = prims::cone(r, h, clamp_smooth(smooth));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_torus(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let r_main = a.take_len(&["radius", "r", "R", "r_main"], 20.0, "ring radius");
        let tube = a.take_len(&["tube", "r2"], 5.0, "tube");
        let smooth = a.take_plain(&["smooth"], 48.0, "smooth");
        if r_main <= 0.0 || tube <= 0.0 {
            self.err_hint(line, "a torus needs a positive radius and tube", "like torus(radius: 2cm, tube: 6mm)");
        }
        let (m, k) = prims::torus(r_main.abs().max(0.5), tube.abs().max(0.2), clamp_smooth(smooth));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_pyramid(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let base = a.take_len(&["base", "b", "s", "size"], 20.0, "base");
        let h = a.take_len(&["h", "height"], 20.0, "height");
        let sides = a.take_plain(&["sides"], 4.0, "sides");
        if base <= 0.0 || h <= 0.0 {
            self.err_hint(line, "pyramid sizes must be positive", "like pyramid(base: 2cm, h: 2cm)");
        }
        let (m, k) = prims::pyramid(base.abs().max(0.5), h.abs().max(0.5), sides.max(3.0) as u32);
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_prism(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let r = a.take_len(&["r", "radius"], 10.0, "radius");
        let h = a.take_len(&["h", "height"], 20.0, "height");
        let sides = a.take_plain(&["sides", "n"], 6.0, "sides");
        if r <= 0.0 || h <= 0.0 {
            self.err_hint(line, "prism sizes must be positive", "like prism(sides: 6, r: 1cm, h: 2cm)");
        }
        let (m, k) = prims::prism(sides.max(3.0).min(64.0) as u32, r.abs().max(0.5), h.abs().max(0.5));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_capsule(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let r = a.take_len(&["r", "radius"], 10.0, "radius");
        let h = a.take_len(&["h", "height", "length"], 30.0, "length");
        let smooth = a.take_plain(&["smooth"], 32.0, "smooth");
        if r <= 0.0 || h < 0.0 {
            self.err_hint(line, "capsule sizes must be positive", "like capsule(r: 1cm, h: 3cm)");
        }
        let (m, k) = prims::capsule(r.abs().max(0.2), h.abs(), clamp_smooth(smooth));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_wedge(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let s = a.take_len(&["s", "size"], 20.0, "size");
        let mut w = s;
        let mut d = s;
        let mut h = s / 2.0;
        if let Some(v) = a.named(&["w", "width"]) { if let Val::Qty(q) = v { w = q.expect_len("width"); } }
        if let Some(v) = a.named(&["d", "depth"]) { if let Val::Qty(q) = v { d = q.expect_len("depth"); } }
        if let Some(v) = a.named(&["h", "height"]) { if let Val::Qty(q) = v { h = q.expect_len("height"); } }
        let (m, k) = prims::wedge(w.abs().max(0.2), d.abs().max(0.2), h.abs().max(0.2));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_plane(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let s = a.take_len(&["s", "size"], 100.0, "size");
        let mut w = s;
        let mut d = s;
        let mut h = 2.0;
        if let Some(v) = a.named(&["w", "width"]) { if let Val::Qty(q) = v { w = q.expect_len("width"); } }
        if let Some(v) = a.named(&["d", "depth"]) { if let Val::Qty(q) = v { d = q.expect_len("depth"); } }
        if let Some(v) = a.named(&["h", "height", "thickness"]) { if let Val::Qty(q) = v { h = q.expect_len("thickness"); } }
        let (m, k) = prims::plane(w.abs().max(0.2), d.abs().max(0.2), h.abs().max(0.5));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_tube(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let r = a.take_len(&["r", "radius"], 5.0, "tube radius");
        let path = match a.take_path(&["path", "points", "through"]) {
            Some(p) if p.len() >= 2 => p,
            _ => {
                return Err(self.err_hint(line, "a tube needs a path of at least 2 points", "like tube(r: 5mm, path: [(0, 0, 0), (0, 5cm, 0)])"))
            }
        };
        let smooth = a.take_plain(&["smooth"], 16.0, "smooth");
        let m = crate::geo::builders::tube(&path, r.abs().max(0.2), clamp_smooth(smooth).min(32));
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_helix(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let radius = a.take_len(&["radius", "r", "R"], 30.0, "radius");
        let pitch = a.take_len(&["pitch"], 10.0, "pitch");
        let turns = a.take_plain(&["turns"], 5.0, "turns");
        let tube = a.take_len(&["tube", "r2"], 5.0, "tube");
        let smooth = a.take_plain(&["smooth"], 16.0, "smooth");
        if radius <= 0.0 || tube <= 0.0 || turns <= 0.0 {
            self.err_hint(line, "a spring needs positive radius, turns and tube", "like helix(radius: 3cm, pitch: 1cm, turns: 5, tube: 5mm)");
        }
        let m = crate::geo::builders::helix(
            radius.abs().max(1.0),
            pitch.abs().max(0.1),
            turns.abs().max(0.1).min(60.0),
            tube.abs().max(0.2),
            clamp_smooth(smooth).min(32),
        );
        self.finish(m, Kind::Generic, &a, line)
    }

    /// OTD3 P2150 — `thread(radius: 3mm, pitch: 1mm, turns: 8)` — the ISO
    /// V-thread ridge that makes a bolt a bolt. Depth defaults to the ISO
    /// 0.541·pitch law; a bare number is read as the diameter (M6 → 6 mm).
    pub(super) fn shape_thread(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let dflt = super::screw::nearest_metric(6.0);
        let radius = a.take_len(&["radius", "r", "R"], dflt.0 / 2.0, "radius");
        let pitch = a.take_len(&["pitch"], dflt.1, "pitch");
        let turns = a.take_plain(&["turns"], 8.0, "turns");
        // depth: explicit, or the ISO V-thread law 0.541·pitch
        let depth = a.take_len(&["depth"], super::screw::thread_depth(pitch), "depth");
        if radius <= 0.0 || pitch <= 0.0 || turns <= 0.0 {
            self.err_hint(line, "a thread needs positive radius, pitch and turns", "like thread(radius: 3mm, pitch: 1mm, turns: 8) — or thread(diameter: 6mm) for an M6");
        }
        let m = crate::geo::builders::thread(
            radius.abs().max(0.5),
            pitch.abs().max(0.05),
            turns.abs().max(0.5).min(200.0),
            depth.abs().max(0.02).min(pitch.abs()),
        );
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_extrude(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let profile = match a.take_profile(&["points", "profile", "polygon"]) {
            Some(p) if p.len() >= 3 => p,
            _ => return Err(self.err_hint(line, "extrude needs a flat shape with 3+ corners", "like extrude [(0,0), (3cm,0), (0,3cm)] depth: 5mm")),
        };
        let depth = a.take_len(&["depth", "height", "h"], 10.0, "depth");
        let _ = a.take_plain(&["smooth"], 0.0, "smooth");
        if depth <= 0.0 {
            self.err_hint(line, format!("depth cannot be negative ({}mm)", depth), format!("did you mean {}mm?", depth.abs()));
        }
        let m = crate::geo::builders::extrude(&profile, depth.abs().max(0.1));
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_revolve(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let profile = match a.take_profile(&["points", "profile"]) {
            Some(p) if p.len() >= 2 => p,
            _ => return Err(self.err_hint(line, "revolve needs a profile of (radius, height) points", "like revolve [(0, 0), (12mm, 0), (12mm, 40mm)]")),
        };
        let angle = a.take_plain(&["angle"], 360.0, "angle");
        let smooth = a.take_plain(&["smooth"], 48.0, "smooth");
        let (m, k) = crate::geo::builders::revolve(&profile, angle, clamp_smooth(smooth));
        self.finish(m, k, &a, line)
    }

    pub(super) fn shape_sweep(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let profile = match a.take_profile(&["profile", "points", "shape"]) {
            Some(p) if p.len() >= 3 => p,
            _ => return Err(self.err_hint(line, "sweep needs a profile of 3+ (x, y) corners", "like sweep [( -1cm, 0), (1cm, 0), (0, 2cm)] path: […]")),
        };
        let path = match a.take_path(&["path", "points", "along"]) {
            Some(p) if p.len() >= 2 => p,
            _ => return Err(self.err_hint(line, "sweep needs a path", "like path: [(0,0,0), (0, 5cm, 0)]")),
        };
        let twist = a.take_plain(&["twist"], 0.0, "twist");
        let m = crate::geo::builders::sweep(&profile, &path, twist);
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_loft(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let sections = match a.take_sections(&["sections", "points"]) {
            Some(s) if s.len() >= 2 => s,
            _ => return Err(self.err_hint(line, "loft needs 2+ shapes with the same corner count", "like loft [ [ (0,0), (2cm,0), (1cm, 2cm) ], [ … ] ]")),
        };
        let spacing = a.take_len(&["spacing", "step"], 10.0, "spacing");
        let _ = a.take_plain(&["smooth"], 0.0, "smooth");
        let m = crate::geo::builders::loft(&sections, spacing.abs().max(0.1));
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_text(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let s = a.take_str(&["text", "value"], "");
        let size = a.take_len(&["size", "height"], 35.0, "text size");
        let depth = a.take_len(&["depth", "d"], 6.0, "text depth");
        if s.is_empty() {
            return Err(self.err_hint(line, "text needs something to say", "like text \"HELLO\""));
        }
        let m = crate::geo::builders::text3d(&s, size.abs().max(1.4), depth.abs().max(0.2));
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_terrain(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let size = a.take_len(&["size", "s"], 100.0, "size");
        let height = a.take_len(&["height", "h"], 20.0, "height");
        let seed = a.take_plain(&["seed"], 7.0, "seed");
        let res = a.take_plain(&["smooth", "resolution", "res"], 32.0, "resolution");
        let m = crate::geo::builders::terrain(size.abs().max(5.0), height.abs().max(0.5), seed.max(0.0) as u64, res.clamp(8.0, 64.0) as u32);
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_metaball(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let path = match a.take_path(&["points", "balls", "at"]) {
            Some(p) if !p.is_empty() => p,
            _ => return Err(self.err_hint(line, "metaball needs ball positions", "like metaball [(0, 0, 0), (2cm, 0, 0)] r: 1cm")),
        };
        let r = a.take_len(&["r", "radius"], 10.0, "ball radius");
        let grid = a.take_plain(&["grid", "resolution"], 24.0, "grid");
        let radii: Vec<f64> = vec![r.abs().max(1.0); path.len()];
        let m = crate::geo::builders::metaballs(&path, &radii, grid.clamp(8.0, 40.0) as u32);
        self.finish(m, Kind::Generic, &a, line)
    }

    pub(super) fn shape_import(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let file = a.take_str(&["file", "from"], "");
        if file.is_empty() {
            return Err(self.err_hint(line, "import needs a file name in quotes", "like import \"cup.stl\""));
        }
        let center = a.take_word(&["center"], "yes") != "no";
        // resolve: examples dir first, then cwd
        let mut path = std::path::PathBuf::from("examples").join(&file);
        if !path.exists() {
            path = std::path::PathBuf::from(&file);
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => {
                return Err(self.err_hint(
                    line,
                    format!("I can't find the file \"{}\"", file),
                    "put it next to your .otd file (or in the examples folder)",
                ))
            }
        };
        let mesh = if file.to_lowercase().ends_with(".stl") {
            crate::geo::builders::import_stl(&bytes)
                .map_err(|e| self.plain_err(line, e))?
        } else if file.to_lowercase().ends_with(".obj") {
            let text = String::from_utf8_lossy(&bytes).to_string();
            crate::geo::builders::import_obj(&text)
                .map_err(|e| self.plain_err(line, e))?
        } else {
            return Err(self.err_hint(line, "import knows .stl and .obj files", "like import \"cup.stl\""));
        };
        let mut mesh = mesh;
        if center {
            // recenter on the origin and rest on the ground
            let bb = mesh.bbox();
            let c = bb.center();
            mesh.transform(&crate::math3::M4::translate(-c.x(), -bb.min.y(), -c.z()));
        }
        self.world.console.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("imported {} — {} triangles", file, mesh.tris.len()),
        });
        self.finish(mesh, Kind::Generic, &a, line)
    }

    // ---------- P1050/P1190: blend + rope (the 2.0 deep-tier words) ----------

    /// `blend(a, b, gap: 8mm)` — SDF smooth-union fillet between two solids.
    pub(super) fn shape_blend(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let mut got: Vec<Val> = Vec::new();
        for arg in args.iter().take(2) {
            // named shape args: blend(a: base, b: knob)
            match arg.name.as_deref() {
                Some("a") | Some("base") | Some("first") => {}
                Some("b") | Some("knob") | Some("second") => {}
                _ => {}
            }
            got.push(self.eval_expr(&arg.val, line)?);
        }
        let mut shapes: Vec<ShapeVal> = Vec::new();
        for v in got {
            match v {
                Val::Shape(s) => shapes.push(s),
                _ => {
                    return Err(self.err_hint(
                        line,
                        "blend wants two objects",
                        "like blend(base, knob, gap: 8mm)",
                    ))
                }
            }
        }
        if shapes.len() < 2 {
            return Err(self.err_hint(line, "blend needs two objects to fuse", "like blend(base, knob, gap: 8mm)"));
        }
        let gap = a.take_len(&["gap", "radius", "r"], 5.0, "gap");
        if gap < 0.0 {
            return Err(self.err_hint(line, "gap cannot be negative", format!("did you mean {}mm?", (-gap * 10.0).round() / 10.0)));
        }
        let res = a.take_plain(&["res", "grid"], 34.0, "resolution").clamp(14.0, 72.0) as u32;
        let merged_a = {
            let mut m = crate::geo::mesh::Mesh::new();
            for x in &shapes[0].meshes {
                m.merge(x);
            }
            m
        };
        let merged_b = {
            let mut m = crate::geo::mesh::Mesh::new();
            for x in &shapes[1].meshes {
                m.merge(x);
            }
            m
        };
        let m = crate::geo::sdf::blend(&merged_a, &merged_b, gap.max(0.05), res);
        // take the first shape's material/color
        let mat = shapes[0].mat;
        let color = shapes[0].color;
        let mut out = ShapeVal { mat, color, ..Default::default() };
        out.meshes.push(m);
        out.kinds.push(Kind::Generic);
        Ok(Val::Shape(out))
    }

    /// `rope(from: (0, 30cm, 0), to: (40cm, 30cm, 0), thickness: 4mm, sag: 5cm)`
    /// — an XPBD-solved hanging cable, swept into a tube.
    pub(super) fn shape_rope(&mut self, args: &[Arg], line: usize) -> Result<Val, crate::lang::errors::Error> {
        let mut a = self.collect_args(args, line)?;
        let from = match a.take_path(&["from", "start"]) {
            Some(p) if !p.is_empty() => p[0],
            _ => {
                return Err(self.err_hint(line, "rope needs a start point", "like rope(from: (0, 30cm, 0), to: (40cm, 30cm, 0))"))
            }
        };
        let to = match a.take_path(&["to", "end"]) {
            Some(p) if !p.is_empty() => p[0],
            _ => {
                return Err(self.err_hint(line, "rope needs an end point", "like rope(from: (0, 30cm, 0), to: (40cm, 30cm, 0))"))
            }
        };
        let thickness = a.take_len(&["thickness", "tube", "r"], 4.0, "thickness");
        if thickness <= 0.0 {
            return Err(self.err_hint(line, "thickness cannot be zero or negative", format!("did you mean {}mm?", (thickness.abs() * 10.0).round().max(1.0) / 10.0)));
        }
        let sag = a.take_len(&["sag"], f64::NAN, "sag");
        let sag_req = if sag.is_nan() { None } else { Some(sag) };
        // gravity: scene setting (m/s²) → engine frame (mm/ms²)
        let g_mm_ms = self.world.gravity * 1e-3;
        let (m, measured) = crate::world::rope::rope(from, to, thickness, sag_req, g_mm_ms);
        self.rope_sag = Some(measured);
        self.finish(m, Kind::Generic, &a, line)
    }
}

fn friendly_positive(v: f64, what: &str) -> (f64, Option<String>) {
    if v < 0.0 {
        (v.abs().max(0.1), Some(format!("did you mean {}mm?", v.abs())))
    } else if v == 0.0 {
        (1.0, Some(format!("{} can't be zero — using 1mm", what)))
    } else {
        (v, None)
    }
}

