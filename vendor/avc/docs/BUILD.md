# Building AVC v3

## Normal build

```bash
cargo build --release        # binaries in target/release/
cargo test                   # 75 tests
```

Rust 1.75+ recommended (developed on 1.98). The four dependencies are pure
Rust; no system libraries, no network at runtime.

## Offline / vendored build

The release archive includes `vendor/` with all dependencies:

```bash
cargo build --release --offline
```

## Binaries

| binary | purpose |
|---|---|
| `avc-demo-synthetic` | 60-frame ground-truth validation + report + all exports |
| `avc-bench` | throughput benchmark per resolution/disparity/path count |
| `avc-process-pair` | process a real image pair into depth/cloud/objects/measurements |

## Using your own stereo pair

1. Capture two **synchronized** rectified-pair images from a calibrated rig.
2. Write `calib.json`:
   ```json
   {"fx": 700, "fy": 700, "cx": 319.5, "cy": 239.5,
    "width": 640, "height": 480, "baseline": 0.12}
   ```
3. `avc-process-pair --left L.png --right R.png --calib calib.json --out out/`

Non-rectified rigs: build a `StereoRig` from the two camera poses in Rust
(`StereoRig { left, right }`); the engine rectifies internally (Fusiello
homographies, epipolar error < 0.35 px).

## Plugging in a neural detector

The `Detection` struct in `avc::perception::detection` is the integration
boundary: replace/augment `detect()` with a YOLO-class detector (ONNX via
the `ort` crate) and feed its boxes/masks through the same lifting +
tracking path. v1/v2 (Python) ship working YOLOv8/YOLOv8-seg ONNX backends
for reference.

## Roadmap

- TSDF + surface nets on the voxel cloud (status: INCOMPLETE)
- loop closure / pose graph (status: INCOMPLETE)
- SIMD acceleration of the SGM aggregation (portable `std::simd`)
- ONNX detector feature (`--features yolo`) using vendored weights
