//! avc-bench (v4): throughput + memory benchmark of every pipeline stage at
//! multiple resolutions, matcher modes (full-res vs pyramid) and end-to-end
//! engine configurations. Writes bench.json + prints a table.
//!
//! v4 additions: pyramid-vs-full matcher comparison, end-to-end engine mode
//! (VO + fusion + detection + tracking included), and peak resident memory
//! (VmHWM from /proc/self/status) per configuration.

use std::time::Instant;

use avc::camera::model::{Intrinsics, StereoRig};
use avc::core::se3::Se3;
use avc::engine::{EngineConfig, VisionEngine};
use avc::sim::scene::SceneSpec;
use avc::stereo::map::SgmParams;
use avc::stereo::sgm::{match_pair_with_buffers, StereoBuffers};

/// Peak resident set size in MB (Linux /proc; 0.0 when unavailable).
fn rss_peak_mb() -> f64 {
    match std::fs::read_to_string("/proc/self/status") {
        Ok(txt) => {
            for line in txt.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0.0);
                    return kb / 1024.0;
                }
            }
            0.0
        }
        Err(_) => 0.0,
    }
}

fn bench_config(w: u32, h: u32, num_disp: usize, paths: usize, pyramid: bool) -> (Intrinsics, SgmParams) {
    let f = 700.0 * w as f64 / 640.0;
    let k = Intrinsics::new(f, f, (w - 1) as f64 / 2.0, (h - 1) as f64 / 2.0, w, h);
    let sgm = SgmParams {
        num_disparities: num_disp,
        disp_min: 2,
        paths,
        pyramid,
        ..Default::default()
    };
    (k, sgm)
}

struct Row {
    resolution: String,
    mode: String,
    stereo_ms: f64,
    fps: f64,
    rss_mb: f64,
    note: String,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let quick = args.iter().any(|a| a == "--quick");

    println!("AVC v{} stereo pipeline benchmark (CPU, {} threads used by rayon)", avc::VERSION, rayon::current_num_threads());
    println!("Synthetic textured scene with sensor noise sigma 2.0; matcher timings are per frame-pair, best of N.\n");

    let rss0 = rss_peak_mb();
    let mut rows: Vec<Row> = Vec::new();

    // ---- matcher micro-benchmarks (reused buffers, like the engine) ----
    let runs: Vec<(u32, u32, usize, usize, bool, usize)> = if quick {
        vec![(320, 240, 48, 4, true, 5)]
    } else {
        vec![
            (320, 240, 48, 4, false, 8),
            (320, 240, 48, 4, true, 8),
            (640, 480, 64, 4, false, 5),
            (640, 480, 64, 4, true, 5),
            (640, 480, 64, 8, true, 5),
        ]
    };
    let mut seen = std::collections::HashSet::new();
    for &(w, h, nd, paths, pyramid, reps) in &runs {
        let key = format!("{w}x{h}/{nd}d/{paths}p/{}", if pyramid { "pyramid" } else { "full" });
        if !seen.insert(key) {
            continue;
        }
        let (k, sgm) = bench_config(w, h, nd, paths, pyramid);
        let scene = SceneSpec::demo();
        let mut bufs = StereoBuffers::new();
        let mut best = f64::INFINITY;
        let mut measured = 0.0;
        for r in 0..=reps {
            let pair = scene.render_frame(r % 60, k, 7000 + r as u64);
            let t0 = Instant::now();
            let disp = match_pair_with_buffers(&pair.left, &pair.right, &sgm, &mut bufs);
            let ms = t0.elapsed().as_secs_f64() * 1e3;
            if r == 0 {
                continue; // warmup
            }
            best = best.min(ms);
            measured = disp.stats.measured_fraction();
        }
        rows.push(Row {
            resolution: format!("{w}x{h}"),
            mode: format!("{}d/{}p/{}", nd, paths, if pyramid { "pyramid" } else { "full" }),
            stereo_ms: best,
            fps: 1000.0 / best,
            rss_mb: rss_peak_mb() - rss0,
            note: format!("measured {measured:.0}%"),
        });
        println!(
            "{:10} {:16} stereo: {:>7.1} ms ({:6.1} FPS)  peak-rss +{:5.1} MB  measured {:.0}%",
            format!("{w}x{h}"),
            format!("{nd}d/{paths}p/{}", if pyramid { "pyramid" } else { "full" }),
            best,
            1000.0 / best,
            rss_peak_mb() - rss0,
            measured * 100.0
        );
    }

    // ---- end-to-end engine mode (stereo + VO + fusion + detect + track) ----
    println!("\nEnd-to-end engine (mean per frame, incl. VO + temporal fusion + detection + tracking):");
    let engine_runs: Vec<(&str, u32, u32)> = if quick {
        vec![("QVGA fast cfg", 320, 240)]
    } else {
        vec![("QVGA fast cfg", 320, 240), ("VGA demo cfg", 640, 480)]
    };
    let mut engine_rows: Vec<Row> = Vec::new();
    for &(label, w, h) in &engine_runs {
        let (k, config) = if w == 640 {
            (config_intrinsics(&EngineConfig::demo()), EngineConfig::demo())
        } else {
            (config_intrinsics(&EngineConfig::fast()), EngineConfig::fast())
        };
        let scene = SceneSpec::demo();
        let rig = scene.rig_at(0, k);
        let Some(mut engine) = VisionEngine::new(&rig, config) else {
            continue;
        };
        let n_frames = 10usize;
        let mut sum = [0.0f64; 8];
        for f in 0..n_frames {
            let pair = scene.render_frame(f % 60, k, 8100 + f as u64);
            let out = engine.process_frame(&pair.left, &pair.right, f, f as f64 / 30.0);
            sum[0] += out.timings.rectify_ms;
            sum[1] += out.timings.stereo_ms;
            sum[2] += out.timings.vo_ms;
            sum[3] += out.timings.lift_ms;
            sum[4] += out.timings.detect_ms;
            sum[5] += out.timings.track_ms;
            sum[6] += out.timings.total_ms;
            sum[7] += out.timings.fusion_ms;
        }
        let n = n_frames as f64;
        let total = sum[6] / n;
        let row = Row {
            resolution: format!("{w}x{h}"),
            mode: label.to_string(),
            stereo_ms: sum[1] / n,
            fps: 1000.0 / total,
            rss_mb: rss_peak_mb(),
            note: format!(
                "vo {:.0} fusion {:.0} lift {:.0} detect {:.0} track {:.0} total {:.0} ms",
                sum[2] / n,
                sum[7] / n,
                sum[3] / n,
                sum[4] / n,
                sum[5] / n,
                total
            ),
        };
        println!(
            "{:10} {:16} mean:  {:>7.1} ms total ({:6.1} FPS)  stereo {:.0} ms  peak-rss {:5.1} MB",
            format!("{w}x{h}"), label, total, 1000.0 / total, sum[1] / n, rss_peak_mb()
        );
        engine_rows.push(row);
    }

    // ---- JSON output ----
    let mut json = String::from("{\n  \"avc_version\": \"");
    json.push_str(avc::VERSION);
    json.push_str("\",\n  \"cpu_threads\": ");
    json.push_str(&rayon::current_num_threads().to_string());
    json.push_str(",\n  \"results\": [\n");
    for (i, r) in rows.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"resolution\": \"{}\", \"mode\": \"{}\", \"stereo_ms\": \"{:.1}\", \"fps\": \"{:.1}\", \"rss_mb\": \"{:.1}\", \"note\": \"{}\"}}{}\n",
            r.resolution, r.mode, r.stereo_ms, r.fps, r.rss_mb, r.note,
            if i + 1 < rows.len() || !engine_rows.is_empty() { "," } else { "" }
        ));
    }
    for (i, r) in engine_rows.iter().enumerate() {
        json.push_str(&format!(
            "    {{\"resolution\": \"{}\", \"mode\": \"engine: {}\", \"stereo_ms\": \"{}\", \"fps\": \"{}\", \"rss_mb\": \"{}\", \"note\": \"{}\"}}{}\n",
            r.resolution, r.mode, format!("{:.1}", r.stereo_ms), format!("{:.1}", r.fps), format!("{:.1}", r.rss_mb), r.note,
            if i + 1 < engine_rows.len() { "," } else { "" }
        ));
    }
    json.push_str("  ]\n}\n");
    std::fs::write("bench.json", json).unwrap();
    println!("\nWrote bench.json. Notes:");
    println!("- matcher rows: full match incl. voting LR check + post filters, reused buffers (engine-identical)");
    println!("- pyramid = dense search at half res + full-res +-3 px refinement (v4 default in the engine);");
    println!("  full = v3-style full-resolution half-pixel grid (kept for the accuracy tests).");
    println!("- engine rows include VO, temporal fusion, detection, tracking; rss is the process VmHWM.");
}

fn config_intrinsics(config: &EngineConfig) -> Intrinsics {
    config.intrinsics.clone()
}
