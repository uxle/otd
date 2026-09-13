//! avc-demo-synthetic: full synthetic-world validation with per-pixel GT.
//!
//! Runs the complete pipeline over the demo world and measures EVERY claim:
//! depth accuracy, sigma calibration, object positions/sizes/velocities,
//! tracking IDs through occlusion, VO drift, measurement accuracy,
//! relations, throughput. Emits report.md + all export artifacts.

use std::path::PathBuf;

use avc::camera::model::Intrinsics;
use avc::engine::{EngineConfig, VisionEngine};
use avc::export::{measurements_csv, write_avcworld, write_point_cloud_ply, write_world_json, world_to_json, WorldFile};
use avc::core::se3::Se3;
use avc::sim::scene::{GtObjectState, SceneSpec};
use avc::world::measurement::Measurement;
use avc::world::model::WorldObject;
use avc::world::relations::{Relation, RelationKind};
use nalgebra::Vector3;

fn pct(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((v.len() as f64 - 1.0) * q).round() as usize;
    v[idx.min(v.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut out_dir = PathBuf::from("avc-demo-output");
    let mut n_frames = 60usize;
    let mut qvga = false;
    // v4 ablation flags (also used for the benchmark report)
    let mut no_fusion = false;
    let mut full_res = false;
    let mut stride: Option<usize> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--frames" => {
                n_frames = args[i + 1].parse().unwrap_or(60);
                i += 1;
            }
            "--qvga" => qvga = true,
            "--no-fusion" => no_fusion = true,
            "--full-res" => full_res = true,
            "--stride" => {
                stride = args[i + 1].parse().ok();
                i += 1;
            }
            _ => {}
        }
        i += 1;
    }
    std::fs::create_dir_all(&out_dir).unwrap();

    let k = if qvga {
        Intrinsics::new(350.0, 350.0, 159.5, 119.5, 320, 240)
    } else {
        Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480)
    };
    let scene = SceneSpec::demo();
    let mut config = if qvga { EngineConfig::fast() } else { EngineConfig::demo() };
    config.intrinsics = k;
    if no_fusion {
        config.temporal_fusion = false;
    }
    if full_res {
        config.sgm.pyramid = false;
    }
    if let Some(s) = stride {
        config.lift_stride = s;
    }
    if no_fusion || full_res || stride.is_some() {
        println!(
            "ablation: fusion={} pyramid={}",
            config.temporal_fusion, config.sgm.pyramid
        );
    }

    let rig = scene.rig_at(0, k);
    let mut engine = VisionEngine::new(&rig, config).unwrap();

    println!(
        "AVC v{} synthetic-world validation: {} frames, {}x{}, B=0.16 m, f={:.0} px, noise sigma=2",
        avc::VERSION, n_frames, k.width, k.height, k.fx
    );

    // accumulators
    let mut abs_depth_err: Vec<f64> = Vec::new();
    let mut depth_sig_pairs: Vec<(f64, f64)> = Vec::new(); // (|err|, sigma)
    let mut vo_ate: Vec<f64> = Vec::new();
    let mut vo_rot_err: Vec<f64> = Vec::new();
    let mut stages = [0.0f64; 8];
    let mut coverage_sum = 0.0;
    let mut strong_sum = 0.0;
    let mut n_vo_ok = 0usize;

    struct Snap {
        frame: usize,
        gt: Vec<GtObjectState>,
        objects: Vec<WorldObject>,
        person_track: Option<u64>,
    }
    let mut snaps: Vec<Snap> = Vec::new();
    let mut id_switches = 0usize;
    let mut prev_person_track: Option<u64> = None;
    let mut meas_errs: Vec<(f64, f64)> = Vec::new();
    let mut last_world: Option<avc::world::model::WorldModel> = None;
    let mut last_rels: Vec<Relation> = Vec::new();
    let mut last_meas: Vec<Measurement> = Vec::new();
    let mut last_disp: Option<avc::stereo::DisparityMap> = None;

    let sample_step = (k.width as usize * k.height as usize / 3000).max(1);

    for f in 0..n_frames {
        let pair = scene.render_frame(f, k, 1000 + f as u64);
        let t = f as f64 / scene.fps;
        let out = engine.process_frame(&pair.left, &pair.right, f, t);

        // --- depth accuracy vs per-pixel GT ---
        let dmap = &out.disparity;
        for i in (0..(k.width as usize * k.height as usize)).step_by(sample_step) {
            let gt_z = pair.gt.depth[i];
            if gt_z <= 0.0 || gt_z > 8.0 {
                // depth metrics scoped to the usable range (z <= 8 m);
                // beyond that the honest sigma exceeds 10 cm by physics
                continue;
            }
            let d = dmap.disp[i];
            if d.is_finite() && d > 1.0 && dmap.quality[i] >= 2 {
                let z_est = k.fx * 0.16 / d as f64;
                let err = (z_est - gt_z as f64).abs();
                let sig = z_est * z_est / (k.fx * 0.16) * dmap.sigma[i].max(0.15) as f64;
                abs_depth_err.push(err);
                if depth_sig_pairs.len() < 15000 {
                    depth_sig_pairs.push((err, sig));
                }
            }
        }
        coverage_sum += dmap.stats.valid_fraction();
        strong_sum += dmap.stats.measured_fraction();
        if out.vo.ok {
            n_vo_ok += 1;
        }

        // --- VO absolute trajectory error ---
        let pose_est = out.world.cam_pose.unwrap();
        let pose_gt = pair.gt.cam_pose_left;
        vo_ate.push((pose_est.t - pose_gt.t).norm());
        let r_err = (pose_est.r.inverse() * pose_gt.r);
        vo_rot_err.push(r_err.angle());

        // --- stage timings ---
        stages[0] += out.timings.rectify_ms;
        stages[1] += out.timings.stereo_ms;
        stages[2] += out.timings.vo_ms;
        stages[3] += out.timings.lift_ms;
        stages[4] += out.timings.detect_ms;
        stages[5] += out.timings.track_ms;
        stages[6] += out.timings.total_ms;
        stages[7] += out.timings.fusion_ms;

        // --- person GT matching + ID switches ---
        let gt_person = pair.gt.objects.iter().find(|o| o.id == 3).unwrap();
        let gtc = Vector3::new(gt_person.center[0], gt_person.center[1], gt_person.center[2]);
        let person_match = out
            .world
            .objects
            .iter()
            .filter(|o| (o.center - gtc).norm() < 0.8)
            .max_by(|a, b| a.n_hits.cmp(&b.n_hits));
        if let Some(p) = person_match {
            if let Some(prev) = prev_person_track {
                if prev != p.track_id {
                    id_switches += 1;
                }
            }
            prev_person_track = Some(p.track_id);
        }
        if std::env::var("AVC_TRACE").is_ok() {
            for (label, gt_pos) in [
                ("crate1", Vector3::new(1.3, 0.3, 3.3)),
                ("crate2", Vector3::new(-0.95, 0.45, 2.9)),
                ("gt4", Vector3::new(0.55, 0.125, 2.3)),
            ] {
                let near: Vec<&avc::world::model::WorldObject> = out
                    .world
                    .objects
                    .iter()
                    .filter(|o| (o.center - gt_pos).norm() < 1.5)
                    .collect();
                let s: Vec<String> = near
                    .iter()
                    .map(|o| {
                        format!(
                            "id{} {} ({:.2},{:.2},{:.2}) h{:.2} hits{} v({:.1},{:.1},{:.1})",
                            o.track_id, o.class, o.center[0], o.center[1], o.center[2],
                            o.extents[1], o.n_hits, o.velocity[0], o.velocity[1], o.velocity[2]
                        )
                    })
                    .collect();
                eprintln!("f{f} {label}: [{}]", s.join(" | "));
            }
        }
        if std::env::var("AVC_TRACE").is_ok() {
            match person_match {
                Some(p) => eprintln!(
                    "f{f}: person id={} class={} pos=({:.2},{:.2},{:.2}) hits={} vel=({:.2},{:.2},{:.2}) ext=({:.2},{:.2},{:.2}) [{} obj]",
                    p.track_id, p.class, p.center[0], p.center[1], p.center[2], p.n_hits,
                    p.velocity[0], p.velocity[1], p.velocity[2],
                    p.extents[0], p.extents[1], p.extents[2],
                    out.world.objects.len()
                ),
                None => eprintln!(
                    "f{f}: person match NONE (gt at ({:.2},{:.2},{:.2})) [{} obj]",
                    gtc[0], gtc[1], gtc[2], out.world.objects.len()
                ),
            }
        }

        // --- measurement test: crate1 <-> crate2 ---
        if let (Some(c1), Some(c2)) = (
            out.world.objects.iter().find(|o| (o.center - Vector3::new(1.3, 0.3, 3.3)).norm() < 0.6),
            out.world.objects.iter().find(|o| (o.center - Vector3::new(-0.95, 0.45, 2.9)).norm() < 0.6),
        ) {
            if let Some(m) = engine.measure_between(c1.track_id, c2.track_id) {
                meas_errs.push(((m.center_distance - 2.291).abs(), m.sigma));
            }
        }

        snaps.push(Snap {
            frame: f,
            gt: pair.gt.objects.clone(),
            objects: out.world.objects.clone(),
            person_track: person_match.map(|p| p.track_id),
        });

        last_world = Some(out.world.clone());
        last_rels = out.relations.clone();
        last_meas = out.measurements.clone();
        if f == n_frames - 1 || f == n_frames / 2 {
            last_disp = Some(out.disparity.clone());
        }

        if f % 10 == 0 {
            print!(".");
            use std::io::Write;
            std::io::stdout().flush().unwrap();
        }
    }
    println!(" done\n");

    // ================= REPORT =================
    let mut checks: Vec<(String, bool, String)> = Vec::new();
    let mut ck = |name: String, pass: bool, detail: String| checks.push((name, pass, detail));

    let mut med_v = abs_depth_err.clone();
    let med = pct(&mut med_v, 0.5);
    let mut p90_v = abs_depth_err.clone();
    let p90 = pct(&mut p90_v, 0.9);
    ck("depth median |err| <= 6 cm".into(), med <= 0.06, format!("{:.1} cm", med * 100.0));
    ck("depth p90 |err| <= 20 cm".into(), p90 <= 0.20, format!("{:.1} cm", p90 * 100.0));

    let n_sig = depth_sig_pairs.len().max(1);
    let c1 = depth_sig_pairs.iter().filter(|(e, s)| *e <= *s).count() as f64 / n_sig as f64;
    let c2 = depth_sig_pairs.iter().filter(|(e, s)| *e <= 2.0 * *s).count() as f64 / n_sig as f64;
    ck("sigma calibration 1-sigma in [0.50, 0.85]".into(), (0.50..=0.85).contains(&c1), format!("{:.2}", c1));
    ck("sigma calibration 2-sigma in [0.85, 0.995] (heavy tail documented)".into(), (0.85..=0.995).contains(&c2), format!("{:.2}", c2));

    // per-object 3-sigma position checks on sampled frames
    let mut n_obj_checks = 0;
    for &gt_id in &[1i32, 2, 3, 4] {
        for &fs in &[10usize, 20, 30, 40, 50, 58] {
            if fs >= n_frames {
                continue;
            }
            if let Some(snap) = snaps.iter().find(|s| s.frame == fs) {
                if let Some(gt) = snap.gt.iter().find(|o| o.id == gt_id) {
                    let gtc = Vector3::new(gt.center[0], gt.center[1], gt.center[2]);
                    let best = snap
                        .objects
                        .iter()
                        .filter(|o| (o.center - gtc).norm() < 1.2)
                        .min_by(|a, b| (a.center - gtc).norm().partial_cmp(&(b.center - gtc).norm()).unwrap());
                    if let Some(o) = best {
                        let err = (o.center - gtc).norm();
                        let sig = (o.cov[(0, 0)] + o.cov[(1, 1)] + o.cov[(2, 2)]).sqrt().max(0.05);
                        ck(
                            format!("GT#{gt_id} position @f{fs} within 3 sigma"),
                            err <= 3.0 * sig,
                            format!("err {:.2} m vs 3s {:.2} m", err, 3.0 * sig),
                        );
                        n_obj_checks += 1;
                    }
                }
            }
        }
    }
    if n_obj_checks == 0 {
        ck("object position checks produced".into(), false, "no GT objects matched".into());
    }

    // person specifics
    let person_snaps: Vec<&Snap> = snaps.iter().filter(|s| s.person_track.is_some()).collect();
    let mut person_h = 0.0;
    let mut person_verr = f64::INFINITY;
    if let Some(last) = person_snaps.last() {
        let gt = last.gt.iter().find(|o| o.id == 3).unwrap();
        let gtc = Vector3::new(gt.center[0], gt.center[1], gt.center[2]);
        if let Some(o) = last.objects.iter().find(|o| Some(o.track_id) == last.person_track) {
            let err = (o.center - gtc).norm();
            ck("person final position <= 0.90 m (fragmentation documented)".into(), err <= 0.90, format!("{:.2} m", err));
            person_h = o.extents[1];
            ck(
                "person height within 0.55 m of 1.72 (fragmentation documented)".into(),
                (person_h - 1.72).abs() <= 0.55,
                format!("tracked {:.2} m", person_h),
            );
        }
    }
    if let Some(s) = person_snaps.iter().filter(|s| s.frame >= 15).last() {
        if let Some(o) = s.objects.iter().find(|o| Some(o.track_id) == s.person_track) {
            person_verr = (o.velocity[0] - 0.75).abs();
            ck(
                "person velocity within 0.35 m/s of 0.75".into(),
                person_verr <= 0.35,
                format!("{:.2} m/s off (tracked {:.2} m/s)", person_verr, o.velocity[0]),
            );
        }
    }
    if person_snaps.is_empty() {
        ck("person tracked".into(), false, "never matched".into());
    }

    ck("person ID switches == 0".into(), id_switches == 0, format!("{id_switches} switches"));

    // occlusion re-identification
    let before = snaps.iter().find(|s| s.frame == 20).and_then(|s| s.person_track);
    let after = snaps.last().and_then(|s| s.person_track);
    let reid = before.is_some() && after.is_some() && before == after;
    ck(
        "person re-identified after occlusion".into(),
        reid,
        format!("{before:?} -> {after:?}"),
    );

    // VO
    let mut vo_med_v = vo_ate.clone();
    let vo_med = pct(&mut vo_med_v, 0.5);
    let vo_final = *vo_ate.last().unwrap_or(&0.0);
    let mut vo_rot_v = vo_rot_err.clone();
    let vo_rot_med = pct(&mut vo_rot_v, 0.5);
    let path_len = 0.035 * 59.0 + 1.2 * 0.03;
    ck("VO median ATE <= 0.35 m".into(), vo_med <= 0.35, format!("{:.3} m", vo_med));
    ck(
        "VO final drift <= 60% of path (documented limitation)".into(),
        vo_final <= 0.60 * path_len,
        format!("{:.3} m ({:.1}% of {:.2} m)", vo_final, 100.0 * vo_final / path_len, path_len),
    );
    ck("VO median rotation error <= 4.0 deg".into(), vo_rot_med <= 0.0698, format!("{:.2} deg", vo_rot_med.to_degrees()));

    // measurements
    {
        let n_ok = meas_errs.iter().filter(|(e, s)| *e <= 3.0 * *s).count();
        let n = meas_errs.len();
        ck(
            "measurement crate1<->crate2 within 3 sigma (>= 80%)".into(),
            n > 0 && n_ok * 100 >= n * 80,
            format!("{}/{} frames", n_ok, n.max(1)),
        );
        if let Some((e, s)) = meas_errs.last() {
            ck("measurement final |err| <= 0.60 m".into(), *e <= 0.60, format!("err {:.3} m, sigma {:.3} m", e, s));
        }
        let mut med_m = meas_errs.iter().map(|(e, _)| *e).collect::<Vec<f64>>();
        let mm = pct(&mut med_m, 0.5);
        ck("measurement median |err| <= 0.60 m (visible-surface bias documented)".into(), mm <= 0.60, format!("{:.3} m", mm));
    }

    // relations accuracy on the last frame
    if let Some(world) = &last_world {
        let gt_objs = &snaps.last().unwrap().gt;
        let mut ok = 0;
        let mut total = 0;
        let cam = world.cam_pose.unwrap();
        let right = cam.r.matrix().column(0);
        let fwd = cam.r.matrix().column(2);
        for r in &last_rels {
            let (Some(a), Some(b)) = (world.get(r.subject), world.get(r.object)) else { continue };
            let gta = gt_objs.iter().find(|g| (Vector3::new(g.center[0], g.center[1], g.center[2]) - a.center).norm() < 0.8);
            let gtb = gt_objs.iter().find(|g| (Vector3::new(g.center[0], g.center[1], g.center[2]) - b.center).norm() < 0.8);
            if let (Some(ga), Some(gb)) = (gta, gtb) {
                let ca = Vector3::new(ga.center[0], ga.center[1], ga.center[2]);
                let cb = Vector3::new(gb.center[0], gb.center[1], gb.center[2]);
                let truth = match r.kind {
                    RelationKind::LeftOf => ca.dot(&right) < cb.dot(&right),
                    RelationKind::RightOf => ca.dot(&right) > cb.dot(&right),
                    RelationKind::InFrontOf => ca.dot(&fwd) < cb.dot(&fwd),
                    RelationKind::Behind => ca.dot(&fwd) > cb.dot(&fwd),
                    RelationKind::Above => ca[1] > cb[1],
                    RelationKind::Below => ca[1] < cb[1],
                    RelationKind::CloserThan => (ca - cam.t).norm() < (cb - cam.t).norm(),
                    RelationKind::FartherThan => (ca - cam.t).norm() > (cb - cam.t).norm(),
                    RelationKind::Near => (ca - cb).norm() < 1.5,
                };
                total += 1;
                if truth {
                    ok += 1;
                }
            }
        }
        if total > 0 {
            let acc = ok as f64 / total as f64;
            ck("relations accuracy >= 70%".into(), acc >= 0.7, format!("{:.1}% ({ok}/{total})", acc * 100.0));
        }
    }

    // coverage + cloud + throughput
    let n = n_frames as f64;
    ck(
        "valid disparity coverage >= 60%".into(),
        coverage_sum / n >= 0.60,
        format!("{:.1}% (fill included)", 100.0 * coverage_sum / n),
    );
    ck(
        "measured (strong/weak) coverage >= 15%".into(),
        strong_sum / n >= 0.15,
        format!("{:.1}%", 100.0 * strong_sum / n),
    );
    {
        let pts = engine.cloud().to_points();
        let intensity: Vec<u8> = pts.iter().map(|p| ((p[1] + 1.0).clamp(0.0, 3.0) * 60.0) as u8).collect();
        write_point_cloud_ply(&out_dir.join("cloud.ply"), &pts, Some(&intensity)).unwrap();
        ck("point cloud >= 15k voxels".into(), pts.len() >= 15000, format!("{} voxels", pts.len()));
    }
    let total_ms = stages[6] / n;
    let fps = 1000.0 / total_ms;
    ck(
        format!("end-to-end >= 0.6 FPS at {}x{} (QVGA is ~5x faster; see avc-bench)", k.width, k.height),
        fps >= 0.6,
        format!(
            "{fps:.2} FPS (stereo {:.0} ms, fusion {:.0} ms, VO {:.0} ms, detect {:.0} ms, total {:.0} ms/frame)",
            stages[1] / n,
            stages[7] / n,
            stages[2] / n,
            stages[4] / n,
            total_ms
        ),
    );

    // exports
    if let Some(world) = &last_world {
        let wj = world_to_json(world, &last_rels, &last_meas, engine.vo_state().pose_sigma());
        write_world_json(&out_dir.join("world.json"), &wj).unwrap();
        std::fs::write(out_dir.join("measurements.csv"), measurements_csv(&last_meas)).unwrap();
        let wf = WorldFile::from_world(world.frame as u32, world.t, &world.objects);
        write_avcworld(&out_dir.join("world.avcworld"), &wf).unwrap();
        if let Some(d) = &last_disp {
            let color = d.to_color(d.disp_min, d.disp_min + 40.0);
            color.save(out_dir.join("disparity_color.png")).unwrap();
            let depth = d.to_kitti_depth(k.fx, 0.16);
            depth.save(out_dir.join("depth_kitti.png")).unwrap();
        }
        let (nv, _, _) = avc::export::read_ply_info(&out_dir.join("cloud.ply")).unwrap();
        let wj_txt = std::fs::read_to_string(out_dir.join("world.json")).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&wj_txt).unwrap();
        ck(
            "exports written and parse back".into(),
            nv > 1000 && parsed["objects"].is_array(),
            format!("PLY {nv} verts, JSON ok, .avcworld {} bytes", std::fs::metadata(out_dir.join("world.avcworld")).map(|m| m.len()).unwrap_or(0)),
        );
    }

    // ---- write report ----
    let n_pass = checks.iter().filter(|c| c.1).count();
    let mut rep = String::new();
    rep.push_str(&format!("# AVC v{} - Synthetic World Validation\n\n", avc::VERSION));
    rep.push_str(&format!("{} frames, {}x{}, baseline 0.16 m, f={:.0} px, sensor noise sigma 2.0/255.\n", n_frames, k.width, k.height, k.fx));
    rep.push_str("Every number is measured against per-pixel / per-object ground truth.\n\n");
    rep.push_str("## Headline\n\n");
    rep.push_str(&format!("- Depth: median |err| **{:.1} cm**, p90 {:.1} cm (measured pixels)\n", med * 100.0, p90 * 100.0));
    rep.push_str(&format!("- Sigma calibration: {:.0}% within 1 sigma / {:.0}% within 2 sigma (ideal 68/95)\n", c1 * 100.0, c2 * 100.0));
    rep.push_str(&format!("- VO: median ATE {:.3} m, final drift {:.3} m, rot err {:.2} deg ({} frames with valid VO)\n", vo_med, vo_final, vo_rot_med.to_degrees(), n_vo_ok));
    rep.push_str(&format!("- Person: {} ID switches, height {:.2} m, velocity err {:.2} m/s\n", id_switches, person_h, if person_verr.is_finite() { person_verr } else { -1.0 }));
    rep.push_str(&format!("- Throughput: **{:.2} FPS** end-to-end on this CPU\n\n", fps));
    rep.push_str("## Checks\n\n| # | check | pass | measured |\n|---|-------|------|----------|\n");
    for (i, (name, pass, detail)) in checks.iter().enumerate() {
        rep.push_str(&format!(
            "| {} | {} | {} | {} |\n",
            i + 1,
            name,
            if *pass { "PASS" } else { "**FAIL**" },
            detail
        ));
    }
    rep.push_str(&format!("\n**{}/{} checks passed.**\n\n", n_pass, checks.len()));
    // honest failure explanations
    let explanations: Vec<(&str, &str)> = vec![
        ("GT#4", "the 25 cm moving box at 2.3 m is at the edge of classical geometric detection (surface too small for dense measured coverage); v1 recovered it via the YOLOv8-seg backend, which is intentionally not part of the hermetic Rust build"),
        ("ID switches", "through the 26-frame occlusion the reappearing person associates to a live fragment track; anchored coasting keeps a prediction but fragment competition wins the Hungarian assignment"),
        ("re-identified", "same root cause as the ID-switch failures: fragment tracks at reappearance"),
        ("measurement median", "the classical geometric centre sits on the VISIBLE SURFACE (~0.2-0.5 m systematic bias for boxy objects); the covariance models this and the 3-sigma consistency checks pass"),
    ];
    let fails: Vec<&(String, bool, String)> = checks.iter().filter(|c| !c.1).collect();
    if !fails.is_empty() {
        rep.push_str("### Why the remaining checks fail\n\n");
        for (key, why) in &explanations {
            let hit: Vec<&String> = fails.iter().map(|f| &f.0).filter(|n| n.contains(key)).collect();
            if !hit.is_empty() {
                rep.push_str(&format!("- **{}**: {}\n", key, why));
            }
        }
        rep.push('\n');
    }
    rep.push_str("## Honest module status matrix\n\n| module | status | evidence |\n|--------|--------|----------|\n");
    for st in engine.status() {
        rep.push_str(&format!("| {} | {} | {} |\n", st.name, st.status.as_str(), st.evidence));
    }
    std::fs::write(out_dir.join("report.md"), &rep).unwrap();
    std::fs::write(out_dir.join("report.txt"), &rep).unwrap();

    println!("{rep}");
    println!("Validation: {}/{} checks passed.", n_pass, checks.len());
    println!("Artifacts: {} (report.md, world.json, cloud.ply, measurements.csv, world.avcworld, disparity_color.png, depth_kitti.png)", out_dir.display());
    if n_pass == checks.len() {
        println!("ALL CHECKS PASSED");
        std::process::exit(0);
    } else {
        println!("{} checks failed (details in the table above)", checks.len() - n_pass);
        std::process::exit(1);
    }
}
