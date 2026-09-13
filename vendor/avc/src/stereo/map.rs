//! Disparity map container, matcher parameters and visualization.

use image::{GrayImage, Rgb, RgbImage};

/// 16-bit grayscale image type (image 0.25 has no GrayImage16 alias).
pub type GrayImage16 = image::ImageBuffer<image::Luma<u16>, Vec<u16>>;

/// Per-pixel match quality class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    Invalid = 0,
    /// Occlusion / mismatch, filled from neighbours (high sigma).
    Filled = 1,
    /// Valid but weak uniqueness margin.
    Weak = 2,
    /// Strong, unique match.
    Strong = 3,
}

/// Full stereo match statistics (honest reporting).
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchStats {
    pub n_pixels: usize,
    pub n_strong: usize,
    pub n_weak: usize,
    pub n_filled: usize,
    pub n_invalid: usize,
    pub n_rejected_uniqueness: usize,
    pub n_rejected_lr: usize,
    pub n_rejected_speckle: usize,
}

impl MatchStats {
    pub fn valid_fraction(&self) -> f64 {
        let v = self.n_strong + self.n_weak + self.n_filled;
        v as f64 / self.n_pixels.max(1) as f64
    }
    pub fn measured_fraction(&self) -> f64 {
        (self.n_strong + self.n_weak) as f64 / self.n_pixels.max(1) as f64
    }
}

/// The output of the stereo matcher.
#[derive(Debug, Clone)]
pub struct DisparityMap {
    pub width: usize,
    pub height: usize,
    /// Disparity in px; `f32::NAN` where invalid.
    pub disp: Vec<f32>,
    /// Disparity sigma in px per pixel.
    pub sigma: Vec<f32>,
    pub quality: Vec<u8>,
    /// v4 pyramid: true where the value is the coarse half-res SGM estimate
    /// (confidence fallback), not a direct full-res census match. Honest map
    /// measurement (sigma_d is widened) but excluded from object geometry:
    /// SGM boundary smearing bridges objects (measured, v4 validation).
    pub coarse: Vec<bool>,
    pub disp_min: f32,
    pub stats: MatchStats,
}

impl DisparityMap {
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.disp[y * self.width + x]
    }
    #[inline]
    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        self.disp[y * self.width + x].is_finite()
    }
    #[inline]
    pub fn quality(&self, x: usize, y: usize) -> Quality {
        match self.quality[y * self.width + x] {
            3 => Quality::Strong,
            2 => Quality::Weak,
            1 => Quality::Filled,
            _ => Quality::Invalid,
        }
    }
    /// Disparity quantised to a KITTI-style 16-bit depth PNG (Z * 256).
    pub fn to_kitti_depth(&self, f: f64, baseline: f64) -> GrayImage16 {
        let mut img = GrayImage16::from_pixel(
            self.width as u32,
            self.height as u32,
            image::Luma([0u16]),
        );
        for y in 0..self.height {
            for x in 0..self.width {
                let d = self.disp[y * self.width + x];
                if d.is_finite() && d > 0.5 {
                    let z = (f * baseline / d as f64 * 256.0).round();
                    let z = z.clamp(0.0, 65535.0) as u16;
                    img.put_pixel(x as u32, y as u32, image::Luma([z]));
                }
            }
        }
        img
    }
    /// Turbo-ish colourmap visualisation (invalid = black).
    pub fn to_color(&self, d_min: f32, d_max: f32) -> RgbImage {
        let mut img = RgbImage::from_pixel(self.width as u32, self.height as u32, Rgb([0, 0, 0]));
        let span = (d_max - d_min).max(1e-3);
        for y in 0..self.height {
            for x in 0..self.width {
                let d = self.disp[y * self.width + x];
                if d.is_finite() {
                    let t = ((d - d_min) / span).clamp(0.0, 1.0);
                    img.put_pixel(x as u32, y as u32, turbo(t as f64));
                }
            }
        }
        img
    }
}

/// Approximate Google Turbo colormap.
pub fn turbo(t: f64) -> Rgb<u8> {
    let t = t.clamp(0.0, 1.0);
    let r = (255.0 * (0.13572138 + t * (4.61539260 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943)))))).clamp(0.0, 255.0);
    let g = (255.0 * (0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604)))))).clamp(0.0, 255.0);
    let b = (255.0 * (0.10667330 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973)))))).clamp(0.0, 255.0);
    Rgb([r as u8, g as u8, b as u8])
}

/// Parameters of the census/SGM matcher (half-pixel slice grid).
#[derive(Debug, Clone)]
pub struct SgmParams {
    /// Number of full-pixel disparities to search (disp_min .. disp_min+n).
    pub num_disparities: usize,
    /// First disparity to search.
    pub disp_min: usize,
    /// SGM P1: penalty for a 1-px disparity jump (halved on the half-grid).
    pub p1: u16,
    /// SGM P2 base: penalty for disparity jumps > 1 px (intensity-adaptive).
    pub p2_base: u16,
    /// Floor for the adaptive P2.
    pub p2_min: u16,
    /// 4 or 8 aggregation paths.
    pub paths: usize,
    /// Uniqueness ratio: second-best cost must exceed best * this.
    pub uniqueness: f32,
    /// Left-right consistency threshold in px.
    pub lr_thresh: f32,
    /// Speckle filter: minimum component size.
    pub speckle_size: usize,
    /// Speckle filter: max disparity difference within a component.
    pub speckle_diff: f32,
    pub do_lr: bool,
    pub do_speckle: bool,
    pub do_fill: bool,
    /// 3x3 median pre-filter on input images.
    pub median_prefilter: bool,
    /// 3x3 median post-filter on the disparity map.
    pub median_postfilter: bool,
    /// v4 pyramid mode: dense search (census + SGM + WTA) runs at HALF
    /// resolution, then a +-3 px local refinement runs at full resolution.
    /// ~4x cheaper dense search, ~6x less peak memory; the accuracy unit
    /// tests exercise the full-resolution matcher, the engine runs the
    /// pyramid (validation re-measures accuracy there).
    pub pyramid: bool,
}

impl Default for SgmParams {
    fn default() -> Self {
        SgmParams {
            num_disparities: 64,
            disp_min: 0,
            p1: 10,
            p2_base: 96,
            p2_min: 12,
            paths: 4,
            uniqueness: 1.1,
            lr_thresh: 1.0,
            speckle_size: 60,
            speckle_diff: 2.0,
            do_lr: true,
            do_speckle: true,
            do_fill: true,
            median_prefilter: true,
            median_postfilter: true,
            pyramid: false,
        }
    }
}
