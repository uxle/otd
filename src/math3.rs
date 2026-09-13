//! P0020 — math core: Vec3, Mat4, AABB. f64, no traits, no dependencies.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct V3(pub [f64; 3]);

impl V3 {
    pub const ZERO: V3 = V3([0.0; 3]);
    pub fn new(x: f64, y: f64, z: f64) -> V3 { V3([x, y, z]) }
    pub fn x(&self) -> f64 { self.0[0] }
    pub fn y(&self) -> f64 { self.0[1] }
    pub fn z(&self) -> f64 { self.0[2] }
    pub fn add(&self, o: &V3) -> V3 { V3([self.0[0]+o.0[0], self.0[1]+o.0[1], self.0[2]+o.0[2]]) }
    pub fn sub(&self, o: &V3) -> V3 { V3([self.0[0]-o.0[0], self.0[1]-o.0[1], self.0[2]-o.0[2]]) }
    pub fn mul(&self, s: f64) -> V3 { V3([self.0[0]*s, self.0[1]*s, self.0[2]*s]) }
    pub fn dot(&self, o: &V3) -> f64 { self.0[0]*o.0[0] + self.0[1]*o.0[1] + self.0[2]*o.0[2] }
    pub fn cross(&self, o: &V3) -> V3 {
        V3([
            self.0[1]*o.0[2] - self.0[2]*o.0[1],
            self.0[2]*o.0[0] - self.0[0]*o.0[2],
            self.0[0]*o.0[1] - self.0[1]*o.0[0],
        ])
    }
    pub fn len(&self) -> f64 { self.dot(self).sqrt() }
    pub fn norm(&self) -> V3 {
        let l = self.len();
        if l < 1e-12 { V3::ZERO } else { self.mul(1.0 / l) }
    }
    pub fn lerp(&self, o: &V3, t: f64) -> V3 { self.add(&o.sub(self).mul(t)) }
    pub fn min(&self, o: &V3) -> V3 { V3([self.0[0].min(o.0[0]), self.0[1].min(o.0[1]), self.0[2].min(o.0[2])]) }
    pub fn max(&self, o: &V3) -> V3 { V3([self.0[0].max(o.0[0]), self.0[1].max(o.0[1]), self.0[2].max(o.0[2])]) }
    pub fn to_f32(&self) -> [f32; 3] { [self.0[0] as f32, self.0[1] as f32, self.0[2] as f32] }
}

/// Column-major 4×4 matrix (m[col*4 + row]).
#[derive(Clone, Copy, Debug)]
pub struct M4(pub [f64; 16]);

impl M4 {
    pub fn identity() -> M4 {
        let mut m = [0.0; 16];
        m[0] = 1.0; m[5] = 1.0; m[10] = 1.0; m[15] = 1.0;
        M4(m)
    }
    pub fn translate(x: f64, y: f64, z: f64) -> M4 {
        let mut m = M4::identity();
        m.0[12] = x; m.0[13] = y; m.0[14] = z;
        m
    }
    pub fn scale(x: f64, y: f64, z: f64) -> M4 {
        let mut m = [0.0; 16];
        m[0] = x; m[5] = y; m[10] = z; m[15] = 1.0;
        M4(m)
    }
    pub fn rot_x(deg: f64) -> M4 { M4::rot_axis(0, deg) }
    pub fn rot_y(deg: f64) -> M4 { M4::rot_axis(1, deg) }
    pub fn rot_z(deg: f64) -> M4 { M4::rot_axis(2, deg) }
    pub fn rot_axis(axis: usize, deg: f64) -> M4 {
        let r = deg.to_radians();
        let (s, c) = r.sin_cos();
        let mut m = [0.0; 16];
        match axis {
            0 => { m[0]=1.0; m[5]=c; m[6]=s; m[9]=-s; m[10]=c; m[15]=1.0; }
            1 => { m[0]=c; m[2]=-s; m[5]=1.0; m[8]=s; m[10]=c; m[15]=1.0; }
            _ => { m[0]=c; m[1]=s; m[4]=-s; m[5]=c; m[10]=1.0; m[15]=1.0; }
        }
        M4(m)
    }
    /// Rotate about arbitrary axis (normalized) by degrees.
    pub fn rot_axis_vec(axis: V3, deg: f64) -> M4 {
        let a = axis.norm();
        let r = deg.to_radians();
        let (s, c) = r.sin_cos();
        let t = 1.0 - c;
        let (x, y, z) = (a.0[0], a.0[1], a.0[2]);
        let m = [
            t*x*x + c,   t*x*y + s*z, t*x*z - s*y, 0.0,
            t*x*y - s*z, t*y*y + c,   t*y*z + s*x, 0.0,
            t*x*z + s*y, t*y*z - s*x, t*z*z + c,   0.0,
            0.0, 0.0, 0.0, 1.0,
        ];
        M4(m)
    }
    /// self * other
    pub fn mul(&self, o: &M4) -> M4 {
        let mut r = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    acc += self.0[k*4 + row] * o.0[col*4 + k];
                }
                r[col*4 + row] = acc;
            }
        }
        M4(r)
    }
    pub fn apply(&self, p: &V3) -> V3 {
        let x = self.0[0]*p.0[0] + self.0[4]*p.0[1] + self.0[8]*p.0[2]  + self.0[12];
        let y = self.0[1]*p.0[0] + self.0[5]*p.0[1] + self.0[9]*p.0[2]  + self.0[13];
        let z = self.0[2]*p.0[0] + self.0[6]*p.0[1] + self.0[10]*p.0[2] + self.0[14];
        V3([x, y, z])
    }
    /// Transform direction (no translation).
    pub fn apply_dir(&self, p: &V3) -> V3 {
        let x = self.0[0]*p.0[0] + self.0[4]*p.0[1] + self.0[8]*p.0[2];
        let y = self.0[1]*p.0[0] + self.0[5]*p.0[1] + self.0[9]*p.0[2];
        let z = self.0[2]*p.0[0] + self.0[6]*p.0[1] + self.0[10]*p.0[2];
        V3([x, y, z])
    }
    /// Transform a normal-like vector using the inverse-transpose (approximate:
    /// fine for rotations; for scales we normalize after).
    pub fn apply_normal(&self, p: &V3) -> V3 {
        // For rotation-only matrices this equals apply_dir; for uniform scale
        // direction is preserved. Non-uniform scale handled by re-deriving
        // normals from geometry after transform (geo::mesh does that).
        self.apply_dir(p)
    }
}

/// Axis-aligned bounding box.
#[derive(Clone, Copy, Debug)]
pub struct Aabb {
    pub min: V3,
    pub max: V3,
}

impl Aabb {
    pub fn empty() -> Aabb { Aabb { min: V3([f64::MAX; 3]), max: V3([f64::MIN; 3]) } }
    pub fn from_center_extent(c: &V3, ext: &V3) -> Aabb {
        Aabb { min: c.sub(ext), max: c.add(ext) }
    }
    pub fn grow_pt(&mut self, p: &V3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    pub fn grow_box(&mut self, o: &Aabb) {
        self.grow_pt(&o.min);
        self.grow_pt(&o.max);
    }
    pub fn center(&self) -> V3 { self.min.add(&self.max).mul(0.5) }
    pub fn size(&self) -> V3 { self.max.sub(&self.min) }
    pub fn radius(&self) -> f64 { self.size().mul(0.5).len() }
    pub fn is_empty(&self) -> bool { self.min.0[0] > self.max.0[0] }
    pub fn intersects(&self, o: &Aabb) -> bool {
        self.min.0[0] <= o.max.0[0] && self.max.0[0] >= o.min.0[0]
            && self.min.0[1] <= o.max.0[1] && self.max.0[1] >= o.min.0[1]
            && self.min.0[2] <= o.max.0[2] && self.max.0[2] >= o.min.0[2]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn vec_basics() {
        let a = V3::new(1.0, 0.0, 0.0);
        let b = V3::new(0.0, 1.0, 0.0);
        assert_eq!(a.cross(&b), V3::new(0.0, 0.0, 1.0));
        assert_eq!(a.dot(&b), 0.0);
    }
    #[test]
    fn mat_translate_rotate() {
        let m = M4::translate(10.0, 0.0, 0.0).mul(&M4::rot_z(90.0));
        let p = m.apply(&V3::new(1.0, 0.0, 0.0));
        // rotate (1,0,0) by 90° about z → (0,1,0), then translate x+10
        assert!((p.x() - 10.0).abs() < 1e-9 && (p.y() - 1.0).abs() < 1e-9);
    }
}
