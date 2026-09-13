//! Built-in modules — the `burn::nn` crate analogue. `Linear`, `Sequential`,
//! `Embedding`, `Dropout`, and activation modules, all on the autodiff tape.

use crate::rng::Rng;

use super::autodiff::{vadd, vdropout, vgather, vmatmul, vrelu, vsigmoid, vtanh, vgelu, Var};
use super::backend::NdArray;
use super::module::Module;
use super::tensor::Tensor;

type Plain = Tensor<NdArray>;

/// Linear layer: y = x·W + b, with Kaiming-uniform weight init (Burn's
/// `LinearConfig::init` uses a uniform by default — so do we).
pub struct Linear {
    pub w: Var, // (in, out)
    pub b: Var, // (out,)
}

impl Linear {
    pub fn new(inp: usize, out: usize, rng: &mut Rng) -> Linear {
        let bound = (1.0 / inp as f32).sqrt();
        let w: Vec<f32> = (0..inp * out).map(|_| rng.next_range(-bound as f64, bound as f64) as f32).collect();
        let b = vec![0.0f32; out];
        Linear {
            w: Var::param(Plain::from_vec(&[inp, out], w)),
            b: Var::param(Plain::from_vec(&[out], b)),
        }
    }
}

impl Module for Linear {
    fn forward(&self, x: &Var) -> Var {
        // x (n, in) · W (in, out) + b (out,)
        let z = vmatmul(x, &self.w);
        // broadcast b over rows: add as (1, out) is fine because vmse/ops
        // are shape-checked — so tile b to (n, out) explicitly.
        let (n, _) = x.val().dims2();
        let bv = self.b.val();
        let out = bv.shape[0];
        let tiled = Plain::from_vec(&[n, out], (0..n * out).map(|i| bv.data[i % out]).collect());
        vadd(&z, &Var::constant(tiled))
    }

    fn params(&self) -> Vec<Var> {
        vec![self.w.clone(), self.b.clone()]
    }

    fn arch(&self) -> String {
        let (i, o) = self.w.val().dims2();
        format!("linear {}→{}", i, o)
    }
}

/// A stack of modules run in order — Burn's `Sequential`.
pub struct Sequential {
    pub layers: Vec<Box<dyn Module>>,
    names: Vec<String>,
}

impl Sequential {
    pub fn new() -> Sequential {
        Sequential { layers: Vec::new(), names: Vec::new() }
    }

    pub fn push(mut self, m: Box<dyn Module>, name: &str) -> Sequential {
        self.layers.push(m);
        self.names.push(name.to_string());
        self
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Var) -> Var {
        let mut cur = x.clone();
        for l in &self.layers {
            cur = l.forward(&cur);
        }
        cur
    }

    fn params(&self) -> Vec<Var> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    fn arch(&self) -> String {
        self.names.join(" → ")
    }
}

/// ReLU module.
pub struct ReLU;
impl Module for ReLU {
    fn forward(&self, x: &Var) -> Var {
        vrelu(x)
    }
    fn params(&self) -> Vec<Var> {
        Vec::new()
    }
    fn arch(&self) -> String {
        "relu".into()
    }
}

/// Tanh module.
pub struct Tanh;
impl Module for Tanh {
    fn forward(&self, x: &Var) -> Var {
        vtanh(x)
    }
    fn params(&self) -> Vec<Var> {
        Vec::new()
    }
    fn arch(&self) -> String {
        "tanh".into()
    }
}

/// Sigmoid module.
pub struct Sigmoid;
impl Module for Sigmoid {
    fn forward(&self, x: &Var) -> Var {
        vsigmoid(x)
    }
    fn params(&self) -> Vec<Var> {
        Vec::new()
    }
    fn arch(&self) -> String {
        "sigmoid".into()
    }
}

/// GELU module (the transformer default).
pub struct Gelu;
impl Module for Gelu {
    fn forward(&self, x: &Var) -> Var {
        vgelu(x)
    }
    fn params(&self) -> Vec<Var> {
        Vec::new()
    }
    fn arch(&self) -> String {
        "gelu".into()
    }
}

/// Embedding table: look up rows by index — the first stop of every
/// language model, now self-made.
pub struct Embedding {
    pub w: Var, // (vocab, dim)
}

impl Embedding {
    pub fn new(vocab: usize, dim: usize, rng: &mut Rng) -> Embedding {
        let std = 1.0 / (vocab as f32).sqrt();
        let t = crate::nn::Tensor::randn(&[vocab, dim], std, rng);
        Embedding { w: Var::param(Plain::from_plain(t)) }
    }

    /// Look up `idx` rows (train-time path — uses the autodiff gather).
    pub fn lookup(&self, idx: &[usize]) -> Var {
        vgather(&self.w, idx)
    }
}

impl Module for Embedding {
    fn forward(&self, x: &Var) -> Var {
        // indices arrive as a (n, 1) value tensor of small integers
        let idx: Vec<usize> = x.val().data.iter().map(|&v| v.max(0.0) as usize).collect();
        self.lookup(&idx)
    }
    fn params(&self) -> Vec<Var> {
        vec![self.w.clone()]
    }
    fn arch(&self) -> String {
        let (v, d) = self.w.val().dims2();
        format!("embedding {}×{}", v, d)
    }
}

/// Dropout module with an explicit train/eval switch (Burn's Dropout).
pub struct Dropout {
    pub p: f32,
    pub train: bool,
    seed: u64,
}

impl Dropout {
    pub fn new(p: f32, seed: u64) -> Dropout {
        Dropout { p, train: true, seed }
    }

    pub fn eval(&mut self) {
        self.train = false;
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Var) -> Var {
        // Rng state can't live behind &self across calls; derive a
        // deterministic per-call seed from the stored seed instead.
        let mut rng = Rng::new(self.seed ^ (x.val().len() as u64).wrapping_mul(0x9E37));
        vdropout(x, self.p, self.train, &mut rng)
    }
    fn params(&self) -> Vec<Var> {
        Vec::new()
    }
    fn arch(&self) -> String {
        format!("dropout p={}", self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::autodiff::{backward, vmse, vmean};

    fn mlp(sizes: &[usize], seed: u64) -> Sequential {
        let mut rng = Rng::new(seed);
        let mut seq = Sequential::new();
        for (i, w) in sizes.windows(2).enumerate() {
            let last = i == sizes.len() - 2;
            seq = seq.push(Box::new(Linear::new(w[0], w[1], &mut rng)), "linear");
            if !last {
                seq = seq.push(Box::new(ReLU), "relu");
            }
        }
        seq
    }

    #[test]
    fn linear_forward_shape() {
        let lin = Linear::new(3, 5, &mut Rng::new(1));
        let x = Var::constant(Plain::from_vec(&[4, 3], vec![1.0; 12]));
        let y = lin.forward(&x);
        assert_eq!(y.val().shape, vec![4, 5]);
    }

    #[test]
    fn sequential_arch_and_params() {
        let net = mlp(&[2, 8, 1], 3);
        assert_eq!(net.arch(), "linear → relu → linear");
        // params: (2×8 + 8) + (8×1 + 1) = 33
        assert_eq!(net.num_params(), 33);
    }

    #[test]
    fn embedding_lookup_gathers_rows() {
        let emb = Embedding::new(10, 4, &mut Rng::new(2));
        let rows = emb.lookup(&[3, 3, 7]);
        assert_eq!(rows.val().shape, vec![3, 4]);
        let v = rows.val().data;
        assert_eq!(v[0..4], v[4..8], "same index must give the same row");
        // gradient flows back to the right rows: vmean over 12 elements ⇒
        // each element 1/12; row 3 is looked up twice ⇒ 2/12, row 7 ⇒ 1/12
        let loss = vmean(&rows);
        backward(&loss);
        let g = emb.w.grad().data;
        assert!((g[3 * 4] - 2.0 / 12.0).abs() < 1e-6, "row 3 grad {}", g[3 * 4]);
        assert!((g[7 * 4] - 1.0 / 12.0).abs() < 1e-6, "row 7 grad {}", g[7 * 4]);
        assert!(g[0..3 * 4].iter().all(|&x| x == 0.0), "rows 0..2 must have zero grad");
    }

    #[test]
    fn mlp_learns_xor_through_burn_api() {
        // the full stack: Sequential + autodiff + Adam through `fit`
        let x = Plain::from_vec(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.]);
        let y = Plain::from_vec(&[4, 1], vec![0., 1., 1., 0.]);
        let ds = crate::burn::data::Supervised::new(x, y);
        let model = mlp(&[2, 16, 16, 1], 9);
        let mut opt = crate::burn::optim::Adam::new(0.05);
        let report = crate::burn::train::fit(&model, &ds, &mut opt, 300, 4, 9, 0);
        assert!(report.final_loss < 0.02, "burn xor loss = {}", report.final_loss);
        // spot-check predictions
        let pred = model.forward(&Var::constant(Plain::from_vec(&[4, 2], vec![0., 0., 0., 1., 1., 0., 1., 1.])));
        let p = pred.val().data;
        assert!(p[0] < 0.1 && p[1] > 0.9 && p[2] > 0.9 && p[3] < 0.1, "preds {:?}", p);
    }

    #[test]
    fn dropout_module_masks_at_train_and_not_eval() {
        let mut d = Dropout::new(0.5, 4);
        let x = Var::constant(Plain::from_vec(&[256], vec![1.0; 256]));
        let y = d.forward(&x);
        let kept = y.val().data.iter().filter(|&&v| v > 0.0).count();
        assert!(kept > 80 && kept < 176, "kept {}", kept);
        d.eval();
        let y2 = d.forward(&x);
        assert_eq!(y2.val().data, x.val().data);
    }

    #[test]
    fn mse_loss_value_matches() {
        let p = Var::constant(Plain::from_vec(&[3], vec![1.0, 2.0, 3.0]));
        let t = Plain::from_vec(&[3], vec![0.0, 0.0, 0.0]);
        let l = vmse(&p, &t);
        assert!((l.val().data[0] - 14.0 / 3.0).abs() < 1e-6);
    }
}
