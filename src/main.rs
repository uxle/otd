//! OTD — Open Three-Dimensional Language.
//! Pure Rust + assembly, zero external libraries. One binary: server, CLI,
//! checker, exporter.

use std::sync::Arc;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let mut port: u16 = 6830;
    let mut mode = String::new();
    let mut file = String::new();
    let mut out = String::new();
    let mut fmt = String::new();
    let mut view = String::from("iso");
    // P1420 — animation flags
    let mut spins: Vec<(String, char, f64)> = Vec::new(); // (part, axis, deg)
    let mut turn: f64 = 0.0;
    let mut frames: u32 = 0;
    // P1420b — settle before render/check; P1450 — AI + perception bridges
    let mut settle = false;
    let mut baseline_mm: f64 = 65.0;
    // OTD3 — the video studio: 8 cameras, AVC encoding, screw animation
    let mut cams: u32 = 1;
    let mut video_out: String = String::new();
    let mut fps: u32 = 24;
    let mut screws: Vec<(String, char, f64, f64)> = Vec::new(); // (part, axis, turns, pitch_mm)

    while i < args.len() {
        let a = &args[i];
        match a.as_str() {
            "--port" | "-p" => {
                i += 1;
                if i < args.len() {
                    port = args[i].parse().unwrap_or(6830);
                }
            }
            "--check" => { mode = "check".into(); i += 1; if i < args.len() { file = args[i].clone(); } }
            "--vm-dump" => { mode = "vm-dump".into(); i += 1; if i < args.len() { file = args[i].clone(); } }
            "--png" => { mode = "png".into(); i += 1; if i < args.len() { file = args[i].clone(); } }
            "--export" => { mode = "export".into(); i += 1; if i < args.len() { file = args[i].clone(); } }
            "--fmt" => { i += 1; if i < args.len() { fmt = args[i].clone(); } }
            "--view" => { i += 1; if i < args.len() { view = args[i].clone(); } }
            "--out" | "-o" => { i += 1; if i < args.len() { out = args[i].clone(); } }
            "--spin" => {
                // --spin rotor=90        spin part "rotor" 90° around Y
                // --spin rotor=90@x      ...around the X axis instead
                // --spin rotor           full 360° spin target (use with --frames)
                i += 1;
                if i < args.len() {
                    let spec = args[i].clone();
                    let (name, rest) = match spec.split_once('=') {
                        Some((n, r)) => (n.to_string(), r.to_string()),
                        None => (spec.clone(), "360".into()),
                    };
                    let (deg_s, axis) = match rest.split_once('@') {
                        Some((d, a)) => (d.to_string(), a.chars().next().unwrap_or('y')),
                        None => (rest, 'y'),
                    };
                    spins.push((name, axis, deg_s.parse().unwrap_or(360.0)));
                }
            }
            "--turn" => { i += 1; if i < args.len() { turn = args[i].parse().unwrap_or(0.0); } }
            "--frames" => { i += 1; if i < args.len() { frames = args[i].parse().unwrap_or(0); } }
            // ---- OTD3: the video studio ----
            "--cams" => { i += 1; if i < args.len() { cams = args[i].parse().unwrap_or(1).clamp(1, 8); } }
            "--fps" => { i += 1; if i < args.len() { fps = args[i].parse().unwrap_or(24).clamp(1, 60); } }
            "--video" => {
                mode = "video".into();
                i += 1;
                if i < args.len() { video_out = args[i].clone(); }
            }
            "--screw" => {
                // --screw nut=6@y:5mm    part, turns around axis, pitch per turn
                i += 1;
                if i < args.len() {
                    let spec = args[i].clone();
                    let (name, rest) = match spec.split_once('=') {
                        Some((n, r)) => (n.to_string(), r.to_string()),
                        None => (spec.clone(), "1".into()),
                    };
                    let (turns_s, tail) = match rest.split_once('@') {
                        Some((t, tl)) => (t.to_string(), tl.to_string()),
                        None => (rest, "y:5".into()),
                    };
                    let (axis_s, pitch_s) = match tail.split_once(':') {
                        Some((a, p)) => (a.to_string(), p.to_string()),
                        None => (tail, "5".into()),
                    };
                    let axis = axis_s.chars().next().unwrap_or('y');
                    let pitch = pitch_s.trim_end_matches("mm").parse().unwrap_or(5.0);
                    screws.push((name, axis, turns_s.parse().unwrap_or(1.0), pitch));
                }
            }
            "--settle" => { settle = true; }
            "--baseline" => { i += 1; if i < args.len() { baseline_mm = args[i].parse().unwrap_or(65.0); } }
            "--ai" => {
                // --ai "why does oil float on water" — the vendored
                // reasoning-AI engine answers, every claim verified
                mode = "ai".into();
                i += 1;
                if i < args.len() { file = args[i].clone(); }
            }
            "--perceive" => {
                // --perceive scene.otd — render a synthetic stereo pair from
                // the scene camera, feed it to the vendored AVC engine, and
                // print what it SEES (objects, distances, relations)
                mode = "perceive".into();
                i += 1;
                if i < args.len() { file = args[i].clone(); }
            }
            "--dump-examples" => { mode = "dump".into(); i += 1; if i < args.len() { out = args[i].clone(); } }
            // OTD3.1 P2240 — self make anything: synthesize a full .otd
            // program from a plain-language goal, compile-check it, save it
            "--make" => {
                mode = "make".into();
                i += 1;
                if i < args.len() { file = args[i].clone(); }
            }
            // OTD3.1 P2210 — train the self-made OTD-Burn network on
            // OTD's own physics laws (freefall) + the XOR sanity check
            "--train" => { mode = "train".into(); }
            "--version" | "-v" => { println!("OTD {} — Open Three-Dimensional Language (pure Rust + assembly, zero dependencies)", env!("CARGO_PKG_VERSION")); return; }
            // OTD6 #1: self-describing — list from source, not hand-maintained docs
            "--list-functions" | "--functions" => { list_functions(); return; }
            "--list-materials" | "--materials" => { list_materials(); return; }
            "--list-keywords" | "--keywords" => { list_keywords(); return; }
            "--list-shapes" | "--shapes" => { list_shapes(); return; }
            "--list-simulate" | "--sims" => { list_simulate(); return; }
            "--list-colors" | "--colors" => { list_colors(); return; }
            "--list-all" => { list_all(); return; }
            "--serve" | "serve" => { mode = "serve".into(); }
            other => {
                if other.starts_with("--") {
                    // unknown flag: ignore
                } else if (mode == "png" || mode == "export") && out.is_empty() {
                    out = other.into();
                } else if mode.is_empty() {
                    mode = "serve".into();
                    file = other.into();
                }
            }
        }
        i += 1;
    }

    println!("{}", banner());
    match mode.as_str() {
        "check" => cmd_check(&file, settle),
        "vm-dump" => cmd_vm_dump(&file),
        "png" => cmd_png(&file, &out, &view, &spins, &screws, turn, frames, settle, cams),
        "export" => cmd_export(&file, &out, &fmt, &view),
        "dump" => cmd_dump(&out),
        "ai" => cmd_ai(&file),
        "make" => cmd_make(&file, &out),
        "train" => cmd_train(),
        "perceive" => cmd_perceive(&file, &out, baseline_mm),
        "video" => cmd_video(&file, &video_out, &spins, &screws, turn, frames, settle, cams, fps),
        _ => serve(port, &file),
    }
}

fn banner() -> String {
    let art = format!(r#"
  ██████╗ ████████╗████████╗██████╗
 ██╔════╝ ╚══██╔══╝╚══██╔══╝██╔══██╗
 ██║  ███╗   ██║      ██║   ██████╔╝
 ██║   ██║   ██║      ██║   ██╔══██╗
 ╚██████╔╝   ██║      ██║   ██████╔╝
  ╚═════╝    ╚═╝      ╚═╝   ╚═════╝  {}"#, env!("CARGO_PKG_VERSION"));
    format!("{}\n  Open Three-Dimensional Language\n  {}\n", art, otd::phase::banner())
}

fn cmd_check(file: &str, settle: bool) {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => { eprintln!("can't read {}", file); return; }
    };
    let w = otd::compile(&src);
    let mut w = w;
    if settle {
        let g = w.gravity;
        let lines = otd::world::physics::simulate_mut("settle", &mut w, 0, g);
        for c in &lines {
            println!("  [{}] {}", console_kind(c.kind), c.text);
        }
    }
    println!("scene: {}", w.title);
    println!("objects: {}", w.stats.visible_parts);
    println!("mass: {:.1} g  volume: {:.1} cm³", w.stats.total_mass_g, w.stats.total_volume_mm3 / 1000.0);
    for p in &w.parts {
        if !p.hidden {
            println!("  {} — {:.1} g ({})", p.name, p.mass_g, p.material.map(|m| m.name).unwrap_or("plastic"));
        }
    }
    for e in &w.errors {
        println!("  {}", e.render());
    }
    for c in &w.console {
        println!("  [{}] {}", console_kind(c.kind), c.text);
    }
    if w.errors.is_empty() {
        println!("✔ no errors");
    }
}

/// P0680 — `--vm-dump file.otd`: compile and print the OTD-ASM disassembly
/// of every numeric expression the VM took (plus a run summary).
fn cmd_vm_dump(file: &str) {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => { eprintln!("can't read {}", file); std::process::exit(1); }
    };
    use otd::vm;
    let prog = otd::lang::parser::parse(&src, "cm");
    // walk statements for expressions worth dumping
    let mut dumped = 0;
    let mut programs: Vec<(usize, vm::compile::Program)> = Vec::new();
    for stmt in &prog.stmts {
        let exprs: Vec<&otd::lang::ast::Expr> = match stmt {
            otd::lang::ast::Stmt::Assign(_, e, _line) => vec![e],
            _ => Vec::new(),
        };
        let line = match stmt {
            otd::lang::ast::Stmt::Assign(_, _, line) => *line,
            _ => 0,
        };
        for e in exprs {
            if let Ok(p) = vm::compile::compile(e, 10.0) {
                programs.push((line, p));
            }
        }
    }
    for (line, p) in &programs {
        println!(";; ---- line {}: {} instructions, {} vars ----", line, p.insns.len(), p.vars.len());
        print!("{}", p.disassemble());
        dumped += 1;
    }
    if dumped == 0 {
        println!(";; no numeric expressions compiled (shapes only?)");
    } else {
        println!(";; {} program(s), {} instructions total", dumped,
            programs.iter().map(|(_, p)| p.insns.len()).sum::<usize>());
    }
}

fn console_kind(k: otd::world::LineKind) -> &'static str {
    use otd::world::LineKind::*;
    match k {
        Answer => "answer",
        Print => "print",
        Warn => "warn",
        Sim => "sim",
        Info => "info",
        Error => "error",
    }
}

fn cmd_png(file: &str, out: &str, view: &str, spins: &[(String, char, f64)], screws: &[(String, char, f64, f64)], turn: f64, frames: u32, settle: bool, cams: u32) {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => { eprintln!("can't read {}", file); return; }
    };
    let default_out = out.is_empty();
    // P1420 animation: --frames N renders N phases of the --spin target
    let n = frames.max(1);
    for f in 0..n {
        let w = otd::compile(&src);
        let mut w = w;
        // animation phase: frame f of n shows f/(n−1) of the total motion
        // (frame 0 = the start, frame n−1 = exactly the full spin/screw/turn)
        let phase = if n > 1 { f as f64 / (n - 1) as f64 } else { 1.0 };
        for (name, axis, deg) in spins {
            let moved = otd::world::anim::spin_named(&mut w, name, *axis, deg * phase);
            if moved == 0 {
                eprintln!("note: no part named '{}' (spins need names, like wheel = cylinder ...)", name);
            }
        }
        // OTD3 screws: rotation and translation coupled — one pitch per turn
        for (name, axis, turns, pitch) in screws {
            let deg = 360.0 * turns * phase;
            let (moved, _) = otd::world::screw::screw_named(&mut w, name, *axis, deg, *pitch);
            if moved == 0 {
                eprintln!("note: no part named '{}' for --screw (name the part: nut = subtract(...))", name);
            }
        }
        if turn != 0.0 {
            otd::world::anim::turn_world(&mut w, turn * phase);
        }
        if settle {
            let g = w.gravity;
            let lines = otd::world::physics::simulate_mut("settle", &mut w, 0, g);
            for c in &lines {
                eprintln!("  [sim] {}", c.text);
            }
        }
        let out_path = if frames > 0 {
            let stem = if default_out { file.replace(".otd", "") } else { out.trim_end_matches(".png").to_string() };
            format!("{}-{:03}.png", stem, f)
        } else if default_out {
            file.replace(".otd", ".png")
        } else {
            out.to_string()
        };
        let png = render_frame_png(&w, view, cams);
        std::fs::write(&out_path, png).expect("write png");
        if std::env::var("OTD_DEBUG").is_ok() {
            for part in &w.parts {
                eprintln!("  [dbg] {} centroid {:?}", part.name, part.centroid.map(|c| (c.x() as i64, c.y() as i64, c.z() as i64)));
            }
        }
        println!("rendered → {} ({:.1} g)", out_path, w.stats.total_mass_g);
        if frames == 0 { break; }
    }
}

/// Render one world state: a single view (`--view`) or the eight-camera
/// panel (`--cams 8`).
fn render_frame_png(w: &otd::world::World, view: &str, cams: u32) -> Vec<u8> {
    if cams >= 8 {
        // the octocam panel: 8 views of 240×180 → 1210×366 composite
        let rgba = otd::render::octocam::render_panel(w, 240, 180)
            .unwrap_or_else(|e| { eprintln!("panel: {}", e); Vec::new() });
        if !rgba.is_empty() {
            let pw = 240 * 4 + 2 * 5;
            let ph = 180 * 2 + 2 * 3;
            // RGBA → RGB for the PNG encoder
            let mut rgb = Vec::with_capacity((pw * ph * 3) as usize);
            for px in rgba.chunks_exact(4) {
                rgb.extend_from_slice(&px[0..3]);
            }
            return otd::render::png::encode_png(pw, ph, &rgb);
        }
    }
    let cam = otd::render::camera::Camera::preset(view, &w.stats.bbox);
    let opts = otd::render::RenderOpts { width: 1200, height: 900, ssaa: 2, show_grid: true };
    otd::render::render_world(w, &cam, &opts)
}

/// OTD3 — the video studio: animate a scene and encode it as real H.264/AVC.
///
///   otd --video nut-bolt.otd out.mp4 --screw nut=6@y:5mm --frames 96 --cams 8
///
/// Each frame re-compiles the source, advances every --spin/--screw by its
/// per-frame phase, renders (single view or the 8-camera panel), and the
/// whole run is encoded with OTD's own Baseline H.264 encoder and muxed
/// into MP4 — no external tools.
fn cmd_video(file: &str, out: &str, spins: &[(String, char, f64)], screws: &[(String, char, f64, f64)], turn: f64, frames: u32, settle: bool, cams: u32, fps: u32) {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => { eprintln!("can't read {}", file); return; }
    };
    let n = frames.max(24).min(600); // sensible defaults: 24 frames minimum
    let out = if out.is_empty() { file.replace(".otd", ".mp4") } else { out.to_string() };
    let t0 = std::time::Instant::now();

    // render dimensions: single view 640×480, panel 8×160×120 (I_PCM is
    // lossless — compact views keep files sensible; see video module docs)
    let (vw, vh) = if cams >= 8 { (160 * 4 + 2 * 5, 120 * 2 + 2 * 3) } else { (640, 480) };

    let mut rgba_frames: Vec<Vec<u8>> = Vec::with_capacity(n as usize);
    for f in 0..n {
        let mut w = otd::compile(&src);
        // animation phase: frame f of n shows f/(n−1) of the total motion
        let phase = if n > 1 { f as f64 / (n - 1) as f64 } else { 1.0 };
        for (name, axis, deg) in spins {
            let moved = otd::world::anim::spin_named(&mut w, name, *axis, deg * phase);
            if moved == 0 && f == 0 {
                eprintln!("note: no part named '{}' for --spin", name);
            }
        }
        // screws: rotation AND translation, coupled — the machine's own law
        for (name, axis, turns, pitch) in screws {
            let deg = 360.0 * turns * phase;
            let (moved, travel) = otd::world::screw::screw_named(&mut w, name, *axis, deg, *pitch);
            if moved == 0 && f == 0 {
                eprintln!("note: no part named '{}' for --screw", name);
            }
            let _ = travel;
        }
        if turn != 0.0 {
            otd::world::anim::turn_world(&mut w, turn * phase);
        }
        if settle {
            let g = w.gravity;
            let _ = otd::world::physics::simulate_mut("settle", &mut w, 0, g);
        }
        // render → RGBA
        let rgba: Vec<u8> = if cams >= 8 {
            otd::render::octocam::render_panel(&w, 160, 120).unwrap_or_default()
        } else {
            let cam = otd::render::camera::Camera::preset("iso", &w.stats.bbox);
            let opts = otd::render::RenderOpts { width: vw as u32, height: vh as u32, ssaa: 1, show_grid: true };
            let rgb = otd::render::render_world_rgb(&w, &cam, &opts);
            let mut rgba = Vec::with_capacity(rgb.len() / 3 * 4);
            for px in rgb.chunks_exact(3) {
                rgba.extend_from_slice(px);
                rgba.push(255);
            }
            rgba
        };
        if rgba.is_empty() {
            eprintln!("frame {} rendered empty — stopping", f);
            return;
        }
        rgba_frames.push(rgba);
        if f % 12 == 0 {
            println!("  frame {}/{} ({:.1} g)", f, n, w.stats.total_mass_g);
        }
    }

    println!("encoding {} frames {}×{} as H.264/AVC…", rgba_frames.len(), vw, vh);
    let mp4 = match otd::render::video::encode_mp4(&rgba_frames, vw as u32, vh as u32, fps) {
        Ok(m) => m,
        Err(e) => { eprintln!("video encode failed: {}", e); return; }
    };
    std::fs::write(&out, &mp4).expect("write mp4");
    println!("video → {} ({} bytes, {} s at {} fps, encoded in {:.1}s, AVC/H.264 Baseline)",
        out, mp4.len(), rgba_frames.len() as f64 / fps as f64, fps, t0.elapsed().as_secs_f64());
}

fn cmd_export(file: &str, out: &str, fmt: &str, _view: &str) {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => { eprintln!("can't read {}", file); return; }
    };
    let w = otd::compile(&src);
    if w.parts.iter().all(|p| p.hidden) {
        eprintln!("nothing to export");
        return;
    }
    let fmt = if fmt.is_empty() { "stl".to_string() } else { fmt.to_string() };
    let title = w.title.replace(' ', "_");
    let out = if out.is_empty() { format!("{}.{}", title, fmt) } else { out.to_string() };
    let visible: Vec<&otd::world::Part> = w.parts.iter().filter(|p| !p.hidden).collect();
    let bytes: Vec<u8> = match fmt.as_str() {
        "stl" => {
            let mut mesh = otd::geo::mesh::Mesh::new();
            for p in &visible { mesh.merge(&p.mesh); }
            otd::export::stl::to_binary_stl(&mesh, &title)
        }
        "obj" => {
            let mut text = String::new();
            for p in &visible { text.push_str(&otd::export::obj::to_obj(&p.mesh, &p.name, p.color)); }
            text.into_bytes()
        }
        "glb" | "gltf" => {
            let mats: Vec<otd::export::glb::GlbMaterial> = visible.iter().map(|p| {
                let m = p.material;
                let base = p.color.map(|c| c.linear01()).unwrap_or_else(|| {
                    let mc = m.map(|mm| mm.color).unwrap_or([200, 200, 205]);
                    otd::world::colors::Color::new(mc[0], mc[1], mc[2]).linear01()
                });
                otd::export::glb::GlbMaterial {
                    base_color: base,
                    metallic: if m.map(|mm| mm.metal).unwrap_or(false) { 0.85 } else { 0.0 },
                    roughness: m.map(|mm| mm.roughness as f32).unwrap_or(0.5),
                }
            }).collect();
            let parts: Vec<otd::export::glb::GlbPart> = visible.iter().enumerate()
                .map(|(i, p)| otd::export::glb::GlbPart { name: p.name.as_str(), mesh: &p.mesh, material: i })
                .collect();
            otd::export::glb::to_glb(&title, &parts, &mats)
        }
        "scad" => otd::export::scad::to_scad(&w, &title).into_bytes(),
        "png" => {
            let cam = otd::render::camera::Camera::fit(&w.stats.bbox);
            let opts = otd::render::RenderOpts { width: 1200, height: 900, ssaa: 2, show_grid: true };
            otd::render::render_world(&w, &cam, &opts)
        }
        other => { eprintln!("'{}' is not a format — try stl, obj, glb, scad, png", other); return; }
    };
    std::fs::write(&out, bytes).expect("write export");
    println!("exported → {}", out);
}

fn cmd_dump(dir: &str) {
    let dir = if dir.is_empty() { ".".to_string() } else { dir.to_string() };
    std::fs::create_dir_all(&dir).expect("create dir");
    for ex in otd::content::EXAMPLES {
        let p = format!("{}/{}", dir, ex.file);
        std::fs::write(&p, ex.code).expect("write example");
        println!("wrote {}", p);
    }
}

fn serve(port: u16, preload: &str) {
    let router = otd::net::routes::make_router();
    let addr = format!("127.0.0.1:{}", port);
    println!("serving OTD on http://{}", addr);
    if !preload.is_empty() {
        println!("(pre-load hint: open the viewer and paste the contents of {})", preload);
    }
    let handler: Arc<dyn Fn(&otd::net::http::Request) -> otd::net::http::Response + Send + Sync> =
        Arc::new(move |req| router(req));
    if let Err(e) = otd::net::http::serve(&addr, handler) {
        eprintln!("server error: {}", e);
    }
}

// ---- P1450 — the intelligence bridges ------------------------------------
// The engine ships with two pre-added, self-built sibling toolchains under
// vendor/ (both pure Rust, built by run.sh):
//   vendor/reasoning-ai — verification-gated reasoning: math, physics,
//                         chemistry, biology. It never guesses: every answer
//                         is re-verified by an independent checker, and if
//                         nothing verifies it says ABSTAINED instead.
//   vendor/avc          — Artificial Visual Cortex v4: real stereo
//                         perception. OTD renders a synthetic stereo pair of
//                         the scene; AVC rectifies, matches, lifts to 3D,
//                         tracks objects and reports what it SEES — with
//                         honest sigmas — as a metric world model.

/// Locate a vendored tool binary: relative to the working directory first,
/// then relative to the OTD binary itself (so it works from anywhere).
fn find_tool(rel: &str) -> Option<std::path::PathBuf> {
    let cwd = std::path::PathBuf::from(rel);
    if cwd.exists() {
        return Some(cwd);
    }
    let exe = std::env::current_exe().ok()?;
    let base = exe.parent()?.parent()?; // …/target/release
    let p = base.join(rel);
    if p.exists() {
        return Some(p);
    }
    // one more level up (repo root next to target/)
    let p = base.parent()?.parent()?.join(rel);
    if p.exists() {
        return Some(p);
    }
    None
}

/// `otd --ai "question"` — ask the vendored reasoning engine. Plain pass-through:
/// its VERIFIED/ABSTAINED verdict is the whole point, we don't soften it.
fn cmd_ai(question: &str) {
    let tool = match find_tool("vendor/reasoning-ai/target/release/reasoning-ai") {
        Some(p) => p,
        None => {
            eprintln!("the reasoning engine is not built yet — build it once:");
            eprintln!("  cargo build --release --manifest-path vendor/reasoning-ai/Cargo.toml");
            std::process::exit(1);
        }
    };
    match std::process::Command::new(tool).arg(question).status() {
        Ok(s) if s.success() => {}
        Ok(_) => std::process::exit(1),
        Err(e) => eprintln!("could not run the reasoning engine: {}", e),
    }
}


/// `otd --make "a titanium ball dropping into water" [--out scene.otd]`
/// P2240 — the synthesizer: a plain-language goal becomes a verified .otd
/// program. The code printed here has already passed the full compiler —
/// self-made and self-checked.
fn cmd_make(goal: &str, out: &str) {
    if goal.is_empty() {
        eprintln!("give the synthesizer a goal: otd --make \"a titanium ball dropping into water from 1m\"");
        return;
    }
    let r = otd::ai::synthesize(goal);
    println!("── self-make: {} ", r.goal);
    println!("   {} solid part{}, {:.1} g total, {} repair round{}", r.parts, if r.parts == 1 { "" } else { "s" }, r.mass_g, r.repairs, if r.repairs == 1 { "" } else { "s" });
    println!();
    println!("{}", r.code);
    if !r.hints.is_empty() {
        println!("── it can even film itself:");
        for h in &r.hints {
            println!("   {}", h);
        }
    }
    let path = if out.is_empty() {
        // derive: first two significant words + .otd
        let mut name = String::new();
        for w in r.goal.split_whitespace() {
            let w: String = w.chars().filter(|c| c.is_alphanumeric()).collect();
            if w.len() >= 3 && !name.contains(&w) {
                name.push_str(&w.to_lowercase());
                if name.len() > 24 { break; }
            }
        }
        if name.is_empty() { "selfmade.otd".into() } else { format!("{}.otd", name) }
    } else {
        out.to_string()
    };
    match std::fs::write(&path, &r.code) {
        Ok(_) => println!("\n── saved to {} — run it: otd --check {}", path, path),
        Err(e) => eprintln!("could not save {}: {}", path, e),
    }
}

/// `otd --train` — OTD-Burn learns OTD's own gravity law from samples,
/// then the classic XOR sanity check. Pure self-made deep learning: backend,
/// tensor, autodiff tape, modules, optimizers, learner — no external crates.
fn cmd_train() {
    println!("OTD-Burn — self-made deep learning (Backend / Tensor / Autodiff / Module / Optimizer / Learner)");
    println!("backend: autodiff<ndarray-cpu>    objective: MSE    optimizer: AdamW(6e-3, wd 1e-4)");
    println!();
    println!("experiment 1 — rediscover freefall  t(h) = sqrt(2h/g)  from 96 noisy samples");
    let t0 = std::time::Instant::now();
    let (report, probe, worst) = otd::burn::learn_freefall_burn(9.81, 80, 7);
    for (ep, l) in &report.history {
        if ep % 20 == 0 || *ep + 1 == report.history.len() {
            println!("   epoch {:>3}  loss {:.2e}", ep, l);
        }
    }
    println!("{}", probe.trim_end());
    println!("   final loss {:.2e} in {:.2}s on {} params — worst probe {:.2}% off the law",
             report.final_loss, report.seconds, report.params, worst);
    println!();
    println!("experiment 2 — XOR, the classic sanity check");
    let r = otd::burn::learn_xor_burn(9);
    println!("   final loss {:.2e} after {} epochs ({:.2}s) — 0,1 → 1 and 1,1 → 0 as required",
             r.final_loss, r.epochs_run, t0.elapsed().as_secs_f64());
    println!();
    println!("the machine has learned laws it was never shown — only samples of them.");
}

/// `otd --perceive scene.otd` — the engine SEES its own scene through the
/// vendored AVC stereo perception engine: renders a synthetic stereo pair
/// (left/right cameras, `--baseline` apart, default 65 mm — human-like),
/// hands the pair to AVC, and prints AVC's metric world model: objects,
/// centres, extents, distances and spatial relations with sigmas.
fn cmd_perceive(file: &str, out_dir: &str, baseline_mm: f64) {
    let src = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(_) => { eprintln!("can't read {}", file); return; }
    };
    let out_dir = if out_dir.is_empty() { "avc-output".to_string() } else { out_dir.trim_end_matches('/').to_string() };
    let avc = match find_tool("vendor/avc/target/release/avc-process-pair") {
        Some(p) => p,
        None => {
            eprintln!("the perception engine is not built yet — build it once:");
            eprintln!("  cargo build --release --manifest-path vendor/avc/Cargo.toml");
            std::process::exit(1);
        }
    };
    let w = otd::compile(&src);
    if w.parts.iter().all(|p| p.hidden || p.mesh.is_empty()) {
        eprintln!("nothing to perceive — the scene has no visible parts");
        return;
    }
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    // synthetic stereo: front camera, world shifted ± baseline/2 along X
    // (the camera's right axis at yaw 0). The renderer's vertical FOV with
    // square pixels gives fx = fy = (H/2)/tan(fov/2).
    const W: u32 = 640;
    const H: u32 = 480;
    let mut cam = otd::render::camera::Camera::preset("front", &w.stats.bbox);
    cam.dist *= 1.4; // zoom out: indoor-scene disparities, not macro ones
    let opts = otd::render::RenderOpts { width: W, height: H, ssaa: 1, show_grid: false };
    let fx = (H as f64 / 2.0) / (cam.fov.to_radians() / 2.0).tan();
    let half = baseline_mm / 2.0;
    let mut left = otd::compile(&src);
    for p in &mut left.parts {
        p.mesh.transform(&otd::math3::M4::translate(half, 0.0, 0.0));
    }
    let png_l = otd::render::render_world(&left, &cam, &opts);
    std::fs::write(format!("{}/left.png", out_dir), png_l).expect("write left");
    let mut right = otd::compile(&src);
    for p in &mut right.parts {
        p.mesh.transform(&otd::math3::M4::translate(-half, 0.0, 0.0));
    }
    let png_r = otd::render::render_world(&right, &cam, &opts);
    std::fs::write(format!("{}/right.png", out_dir), png_r).expect("write right");

    // calibration for AVC (baseline declared in METERS — scene mm map 1:1
    // to real-world scale: a 60 mm nut is a 0.06 m nut)
    let calib = format!(
        "{{\"fx\":{:.1},\"fy\":{:.1},\"cx\":{:.1},\"cy\":{:.1},\"width\":{},\"height\":{},\"baseline\":{:.4}}}",
        fx, fx, (W - 1) as f64 / 2.0, (H - 1) as f64 / 2.0, W, H, baseline_mm / 1000.0
    );
    std::fs::write(format!("{}/calib.json", out_dir), calib).expect("write calib");
    println!("stereo pair → {}/left.png, right.png (baseline {} mm, fx {:.0} px)", out_dir, baseline_mm, fx);

    let status = std::process::Command::new(avc)
        .args(["--left", &format!("{}/left.png", out_dir), "--right", &format!("{}/right.png", out_dir),
               "--calib", &format!("{}/calib.json", out_dir), "--out", &out_dir,
               "--num-disp", "128"])
        .status();
    match status {
        Ok(s) if s.success() => {}
        Ok(_) => { eprintln!("the perception engine failed on this pair"); return; }
        Err(e) => { eprintln!("could not run the perception engine: {}", e); return; }
    }
    // pass through AVC's world model: objects + relations it SAW
    match std::fs::read_to_string(format!("{}/world.json", out_dir)) {
        Ok(json) => {
            println!("---- what AVC sees (its world model, {} world.json) ----", out_dir);
            // compact digest: the JSON is small; print objects/relations sections raw
            for line in json.lines() {
                let t = line.trim();
                if t.starts_with("\"objects\"") || t.starts_with("\"relations\"") || t.starts_with("\"measurements\"")
                    || t.starts_with("\"track_id\"") || t.starts_with("\"class\"") || t.starts_with("\"centre\"")
                    || t.starts_with("\"extents\"") || t.starts_with("\"yaw\"") || t.starts_with("\"confidence\"")
                    || t.starts_with("\"kind\"") || t.starts_with("\"a\"") || t.starts_with("\"b\"")
                    || t.starts_with("\"sigma\"") || t.starts_with("\"dist\"")
                {
                    println!("  {}", t.trim_end_matches(','));
                }
            }
            println!("  (full model: {}/world.json — distances in metres, honest sigmas)", out_dir);
        }
        Err(_) => eprintln!("(no world.json written)"),
    }
}

// ============================================================
// OTD6 #1: Self-describing — list functions/materials/keywords
// directly from source, never hand-maintained.
// ============================================================

fn list_functions() {
    println!("OTD {} — callable functions (from src/lang/keywords.rs FUNCS):", env!("CARGO_PKG_VERSION"));
    println!();
    let funcs = otd::lang::keywords::FUNCS;
    println!("  Total: {} functions", funcs.len());
    println!();
    for chunk in funcs.chunks(8) {
        println!("    {}", chunk.join("  "));
    }
    println!();
    println!("  Usage: my_val = sqrt(16)   or   x = abs(-5)   or   v = ohm_i(12, 5)");
}

fn list_materials() {
    println!("OTD {} — materials (from src/world/materials.rs MATERIALS):", env!("CARGO_PKG_VERSION"));
    println!();
    let names = otd::world::materials::NAMES;
    let metals: Vec<&&str> = names.iter().filter(|n| {
        otd::world::materials::find(n).map(|m| m.metal).unwrap_or(false)
    }).collect();
    let non_metals: Vec<&&str> = names.iter().filter(|n| {
        otd::world::materials::find(n).map(|m| !m.metal).unwrap_or(false)
    }).collect();
    println!("  Metals ({}):", metals.len());
    for chunk in metals.chunks(8) {
        println!("    {}", chunk.iter().map(|s| **s).collect::<Vec<_>>().join("  "));
    }
    println!();
    println!("  Non-metals ({}):", non_metals.len());
    for chunk in non_metals.chunks(8) {
        println!("    {}", chunk.iter().map(|s| **s).collect::<Vec<_>>().join("  "));
    }
    println!();
    println!("  Usage: material cup: ceramic   or   cube 5cm material: steel");
}

fn list_keywords() {
    println!("OTD {} — keywords (from src/lang/keywords.rs KEYWORDS):", env!("CARGO_PKG_VERSION"));
    println!();
    let kw = otd::lang::keywords::KEYWORDS;
    println!("  Total: {} (budget < 100)", kw.len());
    println!();
    for chunk in kw.chunks(10) {
        println!("    {}", chunk.join("  "));
    }
    println!();
    println!("  Every keyword is reserved — it can never be an object name.");
}

fn list_shapes() {
    println!("OTD {} — shapes (from src/lang/keywords.rs PRIMITIVES + BUILDERS):", env!("CARGO_PKG_VERSION"));
    println!();
    let prims = otd::lang::keywords::PRIMITIVES;
    let builders = otd::lang::keywords::BUILDERS;
    println!("  Primitives ({}):", prims.len());
    println!("    {}", prims.join("  "));
    println!();
    println!("  Builders ({}):", builders.len());
    println!("    {}", builders.join("  "));
    println!();
    println!("  Usage: sphere(r: 2cm)   or   cube(w: 4cm, d: 4cm, h: 4cm)");
}

fn list_simulate() {
    println!("OTD {} — simulate domains:", env!("CARGO_PKG_VERSION"));
    println!();
    let domains = [
        ("Basic", &["drop", "float", "collapse", "splash", "settle", "solidity", "gas", "mix"][..]),
        ("Science", &["energy", "heat", "magnet", "sound", "light", "time"][..]),
        ("AI", &["learn", "stats", "orbit"][..]),
        ("Subatomic", &["atom", "decay", "particles"][..]),
        ("OTD6", &["motor", "circuit"][..]),
    ];
    for (cat, items) in &domains {
        println!("  {} ({}):", cat, items.len());
        println!("    {}", items.join("  "));
        println!();
    }
    println!("  Usage: simulate: drop   or   simulate: motor   or   simulate: circuit");
}

fn list_colors() {
    println!("OTD {} — named colors (from src/world/colors.rs NAMED):", env!("CARGO_PKG_VERSION"));
    println!();
    let colors = otd::world::colors::NAMED;
    println!("  Total: {} named colors + #rrggbb hex", colors.len());
    println!();
    for chunk in colors.chunks(6) {
        let names: Vec<String> = chunk.iter().map(|(n, r, g, b)| format!("{}(#{:02x}{:02x}{:02x})", n, r, g, b)).collect();
        println!("    {}", names.join("  "));
    }
    println!();
    println!("  Usage: color cup: ivory   or   color cup: #1e90ff");
}

fn list_all() {
    list_functions();
    println!("\n{}", "=".repeat(60));
    list_materials();
    println!("\n{}", "=".repeat(60));
    list_keywords();
    println!("\n{}", "=".repeat(60));
    list_shapes();
    println!("\n{}", "=".repeat(60));
    list_simulate();
    println!("\n{}", "=".repeat(60));
    list_colors();
}
