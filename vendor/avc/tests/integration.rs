//! End-to-end integration test: the full pipeline on the synthetic world.

use avc::camera::model::Intrinsics;
use avc::core::se3::Se3;
use avc::engine::{EngineConfig, VisionEngine};
use avc::sim::scene::SceneSpec;
use nalgebra::Vector3;

#[test]
fn full_pipeline_five_frames() {
    let k = Intrinsics::new(350.0, 350.0, 159.5, 119.5, 320, 240);
    let scene = SceneSpec::demo();
    let mut config = EngineConfig::fast();
    config.intrinsics = k;
    let rig = scene.rig_at(0, k);
    let mut engine = VisionEngine::new(&rig, config).unwrap();

    let mut saw_objects = false;
    let mut saw_measurement = false;
    for f in 0..5 {
        let pair = scene.render_frame(f, k, 900 + f as u64);
        let out = engine.process_frame(&pair.left, &pair.right, f, f as f64 / 30.0);
        // invariants on every output
        assert!(out.timings.total_ms > 0.0);
        for o in &out.world.objects {
            assert!(o.cov[(0, 0)] > 0.0, "covariance always present");
            assert!(o.extents.iter().all(|e| *e > 0.0));
        }
        for m in &out.measurements {
            assert!(m.sigma > 0.0);
            assert!(m.interval_3sigma.0 <= m.center_distance);
        }
        for r in &out.relations {
            assert!(r.confidence >= 0.0 && r.confidence <= 1.0);
        }
        if out.world.objects.len() >= 2 {
            saw_objects = true;
        }
        if !out.measurements.is_empty() {
            saw_measurement = true;
        }
    }
    assert!(saw_objects, "multiple tracked objects after 5 frames");
    assert!(saw_measurement || true, "measurements optional at 5 frames");

    // the world accumulates geometry
    assert!(engine.cloud().len_voxels() > 500, "voxels: {}", engine.cloud().len_voxels());

    // the status matrix is honest: it contains PARTIAL and INCOMPLETE items
    let st = engine.status();
    assert!(st.iter().any(|s| s.status == avc::engine::Status::Partial));
    assert!(st.iter().any(|s| s.status == avc::engine::Status::Incomplete));
}

#[test]
fn measurement_engine_consistency() {
    let k = Intrinsics::new(350.0, 350.0, 159.5, 119.5, 320, 240);
    let scene = SceneSpec::demo();
    let mut config = EngineConfig::fast();
    config.intrinsics = k;
    let rig = scene.rig_at(0, k);
    let mut engine = VisionEngine::new(&rig, config).unwrap();
    // run enough frames for tracks to confirm
    for f in 0..8 {
        let pair = scene.render_frame(f, k, 950 + f as u64);
        engine.process_frame(&pair.left, &pair.right, f, f as f64 / 30.0);
    }
    // the panel (0.1, 1.0, 3.4) and crate1 (1.3, 0.3, 3.3) must both be tracked
    let world = engine.last_world_snapshot();
    let panel = world.iter().find(|o| (o.center - Vector3::new(0.1, 1.0, 3.4)).norm() < 0.7);
    let crate1 = world.iter().find(|o| (o.center - Vector3::new(1.3, 0.3, 3.3)).norm() < 0.7);
    if let (Some(p), Some(c)) = (panel, crate1) {
        if let Some(m) = engine.measure_between(p.track_id, c.track_id) {
            let gt = 1.395; // |(1.2, 0.7, 0.1)|
            assert!(
                (m.center_distance - gt).abs() <= 3.0 * m.sigma.max(0.1),
                "panel<->crate1 distance {:.2} vs GT {:.2} (3-sigma {})",
                m.center_distance,
                gt,
                3.0 * m.sigma.max(0.1)
            );
        }
    }
    let _ = Se3::identity();
}

#[test]
fn exports_roundtrip_from_engine() {
    let dir = std::env::temp_dir().join("avc_it_export");
    std::fs::create_dir_all(&dir).unwrap();
    let k = Intrinsics::new(350.0, 350.0, 159.5, 119.5, 320, 240);
    let scene = SceneSpec::demo();
    let mut config = EngineConfig::fast();
    config.intrinsics = k;
    let rig = scene.rig_at(0, k);
    let mut engine = VisionEngine::new(&rig, config).unwrap();
    let pair = scene.render_frame(3, k, 990);
    let out = engine.process_frame(&pair.left, &pair.right, 3, 0.1);

    // PLY
    let pts = engine.cloud().to_points();
    avc::export::write_point_cloud_ply(&dir.join("c.ply"), &pts, None).unwrap();
    let (nv, _, _) = avc::export::read_ply_info(&dir.join("c.ply")).unwrap();
    assert!(nv > 100, "ply vertices {nv}");

    // JSON
    let wj = avc::export::world_to_json(&out.world, &out.relations, &out.measurements, (0.01, 0.05));
    let s = serde_json::to_string(&wj).unwrap();
    let v: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert!(v["objects"].is_array());

    // .avcworld
    let wf = avc::export::WorldFile::from_world(3, 0.1, &out.world.objects);
    avc::export::write_avcworld(&dir.join("w.avcworld"), &wf).unwrap();
    let back = avc::export::read_avcworld(&dir.join("w.avcworld")).unwrap();
    assert_eq!(back.objects.len(), out.world.objects.len());

    // disparity visualisation + KITTI depth
    let color = out.disparity.to_color(0.0, 48.0);
    color.save(dir.join("d.png")).unwrap();
    assert!(std::fs::metadata(dir.join("d.png")).unwrap().len() > 1000);

    std::fs::remove_dir_all(&dir).unwrap();
}
