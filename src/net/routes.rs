//! P0630 — REST API + static viewer + caches. All JSON by hand.

use crate::net::base64;
use crate::net::http::{Request, Response};
use crate::net::json::{self, Json};
use crate::render::camera::Camera;
use crate::render::{prepare_lit_scene, render_view, render_world, LitScene, RenderOpts};
use crate::world::eval::World;
use crate::world::eval::{LineKind, Part};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// ---------- caches ----------

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

struct Cache {
    /// code hash → compiled world
    compile: HashMap<u64, Arc<World>>,
    /// code hash → ray-traced lighting for that world (BVH + shadow/AO).
    /// Keyed by the SAME hash as `compile` — it's a pure function of the
    /// geometry, never the camera — so an orbit/zoom drag that keeps
    /// re-rendering the same script hits this cache on every frame after
    /// the first and skips the expensive ray tracing entirely.
    lit: HashMap<u64, Arc<LitScene>>,
    /// (code hash, cam hash, w, h, ssaa) → PNG bytes
    frames: HashMap<u64, Arc<Vec<u8>>>,
}

fn cam_json(w: &World, req_cam: Option<&Json>) -> (Camera, bool) {
    // returns (camera, is_auto_fit)
    let bb = &w.stats.bbox;
    match req_cam {
        Some(j) if !bb.is_empty() && j.get("dist").and_then(|d| d.as_f64()).map(|d| d > 0.0).unwrap_or(false) => {
            let def = Camera::fit(bb);
            let cam = Camera {
                yaw: j.get("yaw").and_then(|v| v.as_f64()).unwrap_or(def.yaw),
                pitch: j.get("pitch").and_then(|v| v.as_f64()).unwrap_or(def.pitch),
                dist: j.get("dist").and_then(|v| v.as_f64()).unwrap_or(def.dist),
                target: {
                    let t = j.get("target").and_then(|v| v.as_arr());
                    match t {
                        Some(a) if a.len() == 3 => crate::math3::V3::new(
                            a[0].as_f64().unwrap_or(0.0),
                            a[1].as_f64().unwrap_or(0.0),
                            a[2].as_f64().unwrap_or(0.0),
                        ),
                        _ => def.target,
                    }
                },
                fov: j.get("fov").and_then(|v| v.as_f64()).unwrap_or(def.fov),
            };
            (cam, false)
        }
        _ => (Camera::fit(bb), true),
    }
}

fn world_stats_json(w: &World) -> Vec<(&'static str, Json)> {
    let s = &w.stats;
    vec![
        ("title", Json::s(&w.title)),
        ("parts", Json::n(s.visible_parts as f64)),
        ("total_parts", Json::n(w.parts.len() as f64)),
        ("mass_g", Json::n(round2(s.total_mass_g))),
        ("volume_mm3", Json::n(round2(s.total_volume_mm3))),
        ("area_mm2", Json::n(round2(s.total_area_mm2))),
        (
            "size",
            if s.bbox.is_empty() {
                Json::Arr(vec![Json::n(0.0); 3])
            } else {
                let sz = s.bbox.size();
                Json::Arr(vec![Json::n(sz.x()), Json::n(sz.y()), Json::n(sz.z())])
            },
        ),
        (
            "com",
            match s.com {
                Some(c) => Json::Arr(vec![Json::n(round1(c.x())), Json::n(round1(c.y())), Json::n(round1(c.z()))]),
                None => Json::Null,
            },
        ),
        (
            "part_list",
            Json::Arr(
                w.parts
                    .iter()
                    .filter(|p| !p.hidden)
                    .map(part_json)
                    .collect(),
            ),
        ),
    ]
}

fn part_json(p: &Part) -> Json {
    Json::obj(vec![
        ("name", Json::s(&p.name)),
        ("mass_g", Json::n(round1(p.mass_g))),
        ("volume_mm3", Json::n(round1(p.volume_mm3))),
        ("material", Json::s(p.material.map(|m| m.name).unwrap_or("plastic"))),
        ("hidden", Json::Bool(p.hidden)),
    ])
}

fn round1(v: f64) -> f64 { (v * 10.0).round() / 10.0 }
fn round2(v: f64) -> f64 { (v * 100.0).round() / 100.0 }

fn console_json(w: &World) -> Json {
    Json::Arr(
        w.console
            .iter()
            .map(|c| {
                Json::obj(vec![
                    ("kind", Json::s(kind_str(c.kind))),
                    ("text", Json::s(&c.text)),
                ])
            })
            .collect(),
    )
}

fn kind_str(k: LineKind) -> &'static str {
    match k {
        LineKind::Answer => "answer",
        LineKind::Print => "print",
        LineKind::Warn => "warn",
        LineKind::Sim => "sim",
        LineKind::Info => "info",
        LineKind::Error => "error",
    }
}

fn errors_json(w: &World) -> Json {
    Json::Arr(
        w.errors
            .iter()
            .map(|e| {
                let mut o = vec![("line", Json::n(e.line as f64)), ("msg", Json::s(&e.msg))];
                if let Some(h) = &e.hint {
                    o.push(("hint", Json::s(h)));
                }
                Json::obj(o)
            })
            .collect(),
    )
}

// ---------- routing ----------

pub fn make_router() -> Arc<dyn Fn(&Request) -> Response + Send + Sync> {
    let cache = Arc::new(Mutex::new(Cache { compile: HashMap::new(), lit: HashMap::new(), frames: HashMap::new() }));
    Arc::new(move |req: &Request| route(req, &cache))
}

fn route(req: &Request, cache: &Mutex<Cache>) -> Response {
    let path = req.path.split('?').next().unwrap_or("/");
    match (req.method.as_str(), path) {
        ("GET", "/") => static_resp("text/html; charset=utf-8", include_str!("../../assets/index.html")),
        ("GET", "/app.js") => static_resp("application/javascript; charset=utf-8", include_str!("../../assets/app.js")),
        ("GET", "/style.css") => static_resp("text/css; charset=utf-8", include_str!("../../assets/style.css")),
        ("GET", "/favicon.svg") => static_resp("image/svg+xml", FAVICON),
        ("GET", "/api/examples") => examples_json(),
        ("GET", "/api/lessons") => lessons_json(),
        ("GET", "/api/keywords") => keywords_json(),
        ("GET", "/health") => Response::json(Json::obj(vec![("ok", Json::Bool(true)), ("engine", Json::s(crate::phase::banner()))]).write()),
        ("POST", "/api/render") => api_render(req, cache),
        ("POST", "/api/export") => api_export(req, cache),
        ("GET", "/") => static_resp("text/html; charset=utf-8", include_str!("../../assets/index.html")),
        _ => Response::not_found(),
    }
}

fn static_resp(ct: &str, body: &str) -> Response {
    Response {
        status: 200,
        content_type: ct.into(),
        body: body.as_bytes().to_vec(),
        extra_headers: vec![("Cache-Control".into(), "max-age=60".into())],
    }
}

const FAVICON: &str = r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><rect width="32" height="32" rx="6" fill="#1a1d24"/><path d="M8 20 L16 8 L24 20 Z" fill="#ffcf5c"/><path d="M8 24 L16 14 L24 24 Z" fill="#7aa2f7"/></svg>"##;

fn examples_json() -> Response {
    let arr: Vec<Json> = crate::content::EXAMPLES
        .iter()
        .map(|e| {
            Json::obj(vec![
                ("file", Json::s(e.file)),
                ("title", Json::s(e.title)),
                ("code", Json::s(e.code)),
            ])
        })
        .collect();
    Response::json(Json::obj(vec![("examples", Json::Arr(arr))]).write())
}

fn lessons_json() -> Response {
    let arr: Vec<Json> = crate::content::LESSONS
        .iter()
        .map(|l| {
            Json::obj(vec![
                ("n", Json::n(l.n as f64)),
                ("title", Json::s(l.title)),
                ("goal", Json::s(l.goal)),
                ("code", Json::s(l.code)),
                ("tryit", Json::s(l.tryit)),
            ])
        })
        .collect();
    Response::json(Json::obj(vec![("lessons", Json::Arr(arr))]).write())
}

fn keywords_json() -> Response {
    let kw = crate::lang::keywords::KEYWORDS;
    Response::json(
        Json::obj(vec![
            ("count", Json::n(kw.len() as f64)),
            ("keywords", Json::Arr(kw.iter().map(|k| Json::s(*k)).collect())),
            (
                "materials",
                Json::Arr(crate::world::materials::NAMES.iter().map(|m| Json::s(*m)).collect()),
            ),
        ])
        .write(),
    )
}

// ---------- render / export ----------

fn get_world(cache: &Mutex<Cache>, code: &str) -> Arc<World> {
    let h = fnv1a(code.as_bytes());
    // Fast path: cache hit, lock held only for the lookup itself.
    if let Some(w) = cache.lock().unwrap().compile.get(&h) {
        return w.clone();
    }
    // Slow path: compile OUTSIDE the lock so one thread's compile doesn't
    // stall every other request's cache lookups/renders on this connection
    // pool (the server is one-thread-per-connection — see net/http.rs).
    // Worst case two threads compile the same new code once each; the
    // cache still converges and both callers get a valid World either way.
    let w = Arc::new(crate::compile(code));
    let mut c = cache.lock().unwrap();
    if c.compile.len() > 16 {
        c.compile.clear(); // simple cap
    }
    c.compile.insert(h, w.clone());
    w
}

/// Same cache-outside-the-lock shape as `get_world`, for the ray-traced
/// lighting: cheap lookup under the lock, expensive `prepare_lit_scene`
/// (BVH build + shadow/AO ray tracing) outside it. `h` is the same
/// content hash used for the compile cache, since the lighting is a pure
/// function of the compiled geometry.
fn get_lit_scene(cache: &Mutex<Cache>, h: u64, world: &World) -> Arc<LitScene> {
    if let Some(l) = cache.lock().unwrap().lit.get(&h) {
        return l.clone();
    }
    let l = Arc::new(prepare_lit_scene(world));
    let mut c = cache.lock().unwrap();
    if c.lit.len() > 16 {
        c.lit.clear();
    }
    c.lit.insert(h, l.clone());
    l
}

fn api_render(req: &Request, cache: &Mutex<Cache>) -> Response {
    let body = match json::parse(&String::from_utf8_lossy(&req.body)) {
        Ok(j) => j,
        Err(e) => return Response::json(err_json(&e)),
    };
    let code = body.get("code").and_then(|c| c.as_str()).unwrap_or("");
    let ssaa = body.get("ssaa").and_then(|v| v.as_f64()).unwrap_or(2.0).clamp(1.0, 2.0) as u32;
    let width = body.get("width").and_then(|v| v.as_f64()).unwrap_or(900.0).clamp(200.0, 1600.0) as u32;
    let height = body.get("height").and_then(|v| v.as_f64()).unwrap_or(600.0).clamp(150.0, 1200.0) as u32;
    let cam_req = body.get("cam");

    let h = fnv1a(code.as_bytes());
    let world = get_world(cache, code);
    let (cam, auto_fit) = cam_json(&world, cam_req);

    // frame cache
    let cam_str = cam_req.map(|j| j.write()).unwrap_or_default();
    let fh = fnv1a(&format!("{}|{}|{}|{}|{}", code, cam_str, width, height, ssaa).as_bytes());
    // Same reasoning as get_world(): look the frame up with a short-lived
    // lock, but rasterize OUTSIDE the lock so one slow render can't stall
    // every other connection's cache access.
    let cached_frame = cache.lock().unwrap().frames.get(&fh).cloned();
    let png: Arc<Vec<u8>> = if let Some(p) = cached_frame {
        p
    } else {
        let opts = RenderOpts { width, height, ssaa, show_grid: true };
        // PERF: on an orbit/zoom drag the code (and so `h`) doesn't change
        // between frames — only `cam` does. get_lit_scene() then skips the
        // ray-traced shadow/AO pass entirely after the first frame of a
        // given script, leaving only the cheap camera-dependent projection
        // and rasterization on the interactive path.
        let lit = get_lit_scene(cache, h, &world);
        let p = Arc::new(render_view(&world, &lit, &cam, &opts));
        let mut c = cache.lock().unwrap();
        if c.frames.len() > 48 {
            c.frames.clear();
        }
        c.frames.insert(fh, p.clone());
        p
    };

    let mut fields = vec![
        ("ok", Json::Bool(world.errors.is_empty() && !world.parts.is_empty())),
        ("png", Json::s(base64::encode(&png))),
        ("stats", Json::obj(world_stats_json(&world))),
        ("console", console_json(&world)),
        ("errors", errors_json(&world)),
        (
            "fit",
            if auto_fit {
                let f = Camera::fit(&world.stats.bbox);
                Json::obj(vec![
                    ("yaw", Json::n(f.yaw)),
                    ("pitch", Json::n(f.pitch)),
                    ("dist", Json::n(round1(f.dist))),
                    ("target", Json::Arr(vec![Json::n(round1(f.target.x())), Json::n(round1(f.target.y())), Json::n(round1(f.target.z()))])),
                ])
            } else {
                Json::Null
            },
        ),
        (
            "cam",
            Json::obj(vec![
                ("yaw", Json::n(cam.yaw)),
                ("pitch", Json::n(cam.pitch)),
                ("dist", Json::n(round1(cam.dist))),
                ("target", Json::Arr(vec![Json::n(round1(cam.target.x())), Json::n(round1(cam.target.y())), Json::n(round1(cam.target.z()))])),
            ]),
        ),
        ("gravity", Json::n(world.gravity)),
    ];
    let _ = &mut fields;
    Response::json(Json::obj(fields).write())
}

fn err_json(msg: &str) -> String {
    Json::obj(vec![
        ("ok", Json::Bool(false)),
        ("errors", Json::Arr(vec![Json::obj(vec![("line", Json::n(0.0)), ("msg", Json::s(msg))])])),
    ])
    .write()
}

fn api_export(req: &Request, cache: &Mutex<Cache>) -> Response {
    let body = match json::parse(&String::from_utf8_lossy(&req.body)) {
        Ok(j) => j,
        Err(e) => return Response::json(err_json(&e)),
    };
    let code = body.get("code").and_then(|c| c.as_str()).unwrap_or("");
    let fmt = body.get("fmt").and_then(|c| c.as_str()).unwrap_or("stl").to_lowercase();
    let world = get_world(cache, code);
    if world.parts.iter().all(|p| p.hidden || p.mesh.is_empty()) {
        return Response::json(err_json("nothing to export — make something first"));
    }
    let title = world.title.replace(' ', "_");
    let visible: Vec<&Part> = world.parts.iter().filter(|p| !p.hidden && !p.mesh.is_empty()).collect();
    match fmt.as_str() {
        "stl" => {
            // merge visible parts into one watertight-per-part… STL is single solid:
            // export the first part if one, else merged soup
            let mut mesh = crate::geo::mesh::Mesh::new();
            for p in &visible {
                mesh.merge(&p.mesh);
            }
            let bytes = crate::export::stl::to_binary_stl(&mesh, &title);
            Response::binary("model/stl", bytes, &format!("{}.stl", title))
        }
        "obj" => {
            let mut text = String::new();
            for p in &visible {
                text.push_str(&crate::export::obj::to_obj(&p.mesh, &p.name, p.color));
            }
            Response::binary("text/plain", text.into_bytes(), &format!("{}.obj", title))
        }
        "glb" | "gltf" => {
            let _parts: Vec<crate::export::glb::GlbPart> = visible
                .iter()
                .map(|p| crate::export::glb::GlbPart { name: p.name.as_str(), mesh: &p.mesh, material: 0 })
                .collect();
            let mats: Vec<crate::export::glb::GlbMaterial> = visible
                .iter()
                .map(|p| {
                    let m = p.material;
                    let base = p
                        .color
                        .map(|c| c.linear01())
                        .unwrap_or_else(|| {
                            let mc = m.map(|mm| mm.color).unwrap_or([200, 200, 205]);
                            crate::world::colors::Color::new(mc[0], mc[1], mc[2]).linear01()
                        });
                    crate::export::glb::GlbMaterial {
                        base_color: base,
                        metallic: if m.map(|mm| mm.metal).unwrap_or(false) { 0.85 } else { 0.0 },
                        roughness: m.map(|mm| mm.roughness as f32).unwrap_or(0.5),
                    }
                })
                .collect();
            // each part references its own material index
            let parts: Vec<crate::export::glb::GlbPart> = visible
                .iter()
                .enumerate()
                .map(|(i, p)| crate::export::glb::GlbPart { name: p.name.as_str(), mesh: &p.mesh, material: i })
                .collect();
            let bytes = crate::export::glb::to_glb(&title, &parts, &mats);
            Response::binary("model/gltf-binary", bytes, &format!("{}.glb", title))
        }
        "scad" => {
            let text = crate::export::scad::to_scad(&world, &title);
            Response::binary("text/plain", text.into_bytes(), &format!("{}.scad", title))
        }
        "png" => {
            let cam = Camera::fit(&world.stats.bbox);
            let opts = RenderOpts { width: 1200, height: 900, ssaa: 2, show_grid: true };
            let bytes = render_world(&world, &cam, &opts);
            Response::binary("image/png", bytes, &format!("{}.png", title))
        }
        other => Response::json(err_json(&format!(
            "'{}' is not an export format — try stl, obj, glb, scad, or png",
            other
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fnv_hash_stable() {
        assert_eq!(fnv1a(b"cup"), fnv1a(b"cup"));
        assert_ne!(fnv1a(b"cup"), fnv1a(b"mug"));
    }

    #[test]
    fn render_api_shape() {
        let cache = Mutex::new(Cache { compile: HashMap::new(), lit: HashMap::new(), frames: HashMap::new() });
        let req = Request {
            method: "POST".into(),
            path: "/api/render".into(),
            headers: Vec::new(),
            body: br#"{"code":"cube 2cm"}"#.to_vec(),
        };
        let resp = api_render(&req, &cache);
        assert_eq!(resp.status, 200);
        let j = json::parse(&String::from_utf8_lossy(&resp.body)).unwrap();
        assert!(j.get("ok").unwrap() == &Json::Bool(true));
        assert!(j.get("png").unwrap().as_str().unwrap().len() > 100);
        let stats = j.get("stats").unwrap();
        assert!(stats.get("mass_g").unwrap().as_f64().unwrap() > 0.0);
    }
}
