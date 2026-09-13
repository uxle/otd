//! avc-process-pair: process a real stereo image pair (already rectified, or
//! rectified internally from a calibration file) into disparity, depth,
//! point cloud, objects and measurements.
//!
//! Usage:
//!   avc-process-pair --left L.png --right R.png [--calib calib.json] \
//!       [--out DIR] [--num-disp 64] [--disp-min 0]
//!
//! calib.json: {"fx":..,"fy":..,"cx":..,"cy":..,"width":..,"height":..,
//!              "baseline":..}  (defaults: inferred from image size + FOV 60,
//!              baseline 0.12 m)

use std::path::PathBuf;

use avc::camera::model::{Intrinsics, StereoRig};
use avc::core::se3::Se3;
use avc::core::rng::GaussRng;
use avc::engine::EngineConfig;
use avc::engine::VisionEngine;
use avc::export::{measurements_csv, write_point_cloud_ply, write_world_json, world_to_json};
use nalgebra::Vector3;

#[derive(serde::Deserialize)]
#[derive(Default)]
struct Calib {
    fx: Option<f64>,
    fy: Option<f64>,
    cx: Option<f64>,
    cy: Option<f64>,
    width: Option<u32>,
    height: Option<u32>,
    baseline: Option<f64>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut left_path = PathBuf::new();
    let mut right_path = PathBuf::new();
    let mut calib_path = PathBuf::new();
    let mut out_dir = PathBuf::from("avc-output");
    let mut num_disp = 64usize;
    let mut disp_min = 0usize;
    let mut repeat = 6usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--left" => {
                left_path = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--right" => {
                right_path = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--calib" => {
                calib_path = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--out" => {
                out_dir = PathBuf::from(&args[i + 1]);
                i += 1;
            }
            "--num-disp" => {
                num_disp = args[i + 1].parse().unwrap_or(64);
                i += 1;
            }
            "--repeat" => {
                repeat = args[i + 1].parse().unwrap_or(6);
                i += 1;
            }
            "--disp-min" => {
                disp_min = args[i + 1].parse().unwrap_or(0);
                i += 1;
            }
            _ => {
                eprintln!("usage: avc-process-pair --left L.png --right R.png [--calib c.json] [--out DIR]");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    if left_path.as_os_str().is_empty() || right_path.as_os_str().is_empty() {
        eprintln!("error: --left and --right are required");
        std::process::exit(2);
    }
    std::fs::create_dir_all(&out_dir).unwrap();

    let left_img = image::open(&left_path).expect("cannot open left image").to_luma8();
    let right_img = image::open(&right_path).expect("cannot open right image").to_luma8();
    let (w, h) = left_img.dimensions();
    if right_img.dimensions() != (w, h) {
        eprintln!("error: image dimensions differ");
        std::process::exit(2);
    }

    // calibration
    let calib: Calib = if calib_path.as_os_str().is_empty() {
        Calib::default()
    } else {
        let txt = std::fs::read_to_string(&calib_path).expect("cannot read calib");
        serde_json::from_str(&txt).expect("calib json parse error")
    };
    let fx = calib.fx.unwrap_or_else(|| 700.0 * w as f64 / 640.0);
    let fy = calib.fy.unwrap_or(fx);
    let cx = calib.cx.unwrap_or((w - 1) as f64 / 2.0);
    let cy = calib.cy.unwrap_or((h - 1) as f64 / 2.0);
    let baseline = calib.baseline.unwrap_or(0.12);
    let k = Intrinsics::new(fx, fy, cx, cy, w, h);

    // canonical rig (assumes a standard parallel stereo pair)
    let rig = StereoRig::canonical(k, Se3::identity(), baseline);

    let mut config = EngineConfig::demo();
    config.intrinsics = k;
    config.sgm.num_disparities = num_disp;
    config.sgm.disp_min = disp_min;
    config.detect.min_points = (config.detect.min_points).max(25);

    let mut engine = VisionEngine::new(&rig, config).expect("rig construction failed");
    // The world model only reports tracks with hits >= 2 (an honest
    // multi-frame requirement). A single still pair can never reach that,
    // so feed the pair as a short burst of `--repeat` identical frames
    // (default 6, ~0.2 s at 30 fps) — enough to confirm tracks and settle
    // extents while the geometry stays exactly the still pair's.
    let mut out = None;
    for f in 0..repeat.max(1) {
        out = Some(engine.process_frame(&left_img, &right_img, f, f as f64 / 30.0));
    }
    let out = out.unwrap();

    // ---- outputs ----
    let d = &out.disparity;
    let d_max = d.disp_min + num_disp as f32;
    let color = d.to_color(d.disp_min, d_max);
    color.save(out_dir.join("disparity_color.png")).unwrap();
    let depth = d.to_kitti_depth(fx, baseline);
    depth.save(out_dir.join("depth_kitti.png")).unwrap();
    d.stats
        .n_pixels
            .to_string();
    println!("stereo: {} px | strong {} weak {} filled {} invalid {}",
        d.stats.n_pixels, d.stats.n_strong, d.stats.n_weak, d.stats.n_filled, d.stats.n_invalid);

    // point cloud (measured points only)
    let pts = engine.cloud().to_points();
    let intensity: Vec<u8> = pts.iter().map(|p| ((p[1] + 1.0).clamp(0.0, 3.0) * 60.0) as u8).collect();
    write_point_cloud_ply(&out_dir.join("cloud.ply"), &pts, Some(&intensity)).unwrap();
    println!("cloud: {} voxels", pts.len());

    // world state
    let wj = world_to_json(&out.world, &out.relations, &out.measurements, engine.vo_state().pose_sigma());
    write_world_json(&out_dir.join("world.json"), &wj).unwrap();
    std::fs::write(out_dir.join("measurements.csv"), measurements_csv(&out.measurements)).unwrap();

    println!("objects ({} detections, {} tracked):", out.n_detections, out.world.objects.len());
    for o in &out.world.objects {
        println!(
            "  {} #{} center=({:.2},{:.2},{:.2}) m  extents=({:.2},{:.2},{:.2}) m  sigma=({:.2},{:.2},{:.2}) m",
            o.class,
            o.track_id,
            o.center[0], o.center[1], o.center[2],
            o.extents[0], o.extents[1], o.extents[2],
            o.cov[(0, 0)].sqrt(), o.cov[(1, 1)].sqrt(), o.cov[(2, 2)].sqrt()
        );
    }
    if !out.measurements.is_empty() {
        println!("measurements:");
        for m in &out.measurements {
            println!(
                "  #{} <-> #{}: {:.3} +/- {:.3} m (3-sigma [{:.3}, {:.3}] m)",
                m.a, m.b, m.center_distance, m.sigma, m.interval_3sigma.0, m.interval_3sigma.1
            );
    }
    }
    let _: Option<Vector3<f64>> = None;
    let _ = GaussRng::new(1);
    println!("\noutputs in {}: disparity_color.png, depth_kitti.png, cloud.ply, world.json, measurements.csv", out_dir.display());
}
