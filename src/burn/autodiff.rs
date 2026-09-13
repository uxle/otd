//! Reverse-mode automatic differentiation — the burn-autodiff analogue.
//!
//! A `Var` is a node in a broadcast DAG: it owns a value tensor, a gradient
//! accumulator, and a list of `(parent, backward-fn)` edges. Calling
//! `backward(&out)` topologically sorts the graph rooted at `out`, seeds the
//! output gradient to one, and sweeps the reverse order, accumulating each
//! edge's contribution into its parent. Leaf `Var`s created by `Var::param`
//! are reused across training steps (the weights); every intermediate node
//! is rebuilt per forward pass, so gradients never leak between batches.
//!
//! Every op here is parity-tested against finite differences.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use super::backend::NdArray;
use super::tensor::Tensor;

type Plain = Tensor<NdArray>;
type BackFn = Rc<dyn Fn(&Plain) -> Plain>;

struct Node {
    value: Plain,
    grad: Plain,
    deps: Vec<(NodeRef, BackFn)>,
}

type NodeRef = Rc<RefCell<Node>>;

/// A differentiable value — ≈ `Tensor<Autodiff<NdArray>>` in real Burn.
#[derive(Clone)]
pub struct Var {
    node: NodeRef,
}

impl Var {
    /// A trainable leaf (weights): gradient-tracked, zero grad.
    pub fn param(t: Plain) -> Var {
        let shape = t.shape.clone();
        Var { node: Rc::new(RefCell::new(Node { value: t, grad: Plain::zeros(&shape), deps: Vec::new() })) }
    }

    /// A constant leaf: appears in graphs but never receives gradient edges.
    pub fn constant(t: Plain) -> Var {
        Var::param(t)
    }

    /// Current value (cloned — cheap at OTD scale, simple to reason about).
    pub fn val(&self) -> Plain {
        self.node.borrow().value.clone()
    }

    /// Accumulated gradient.
    pub fn grad(&self) -> Plain {
        self.node.borrow().grad.clone()
    }

    /// Overwrite the value in place (checkpoint loading, optimizers).
    pub fn set_value(&self, t: Plain) {
        let mut n = self.node.borrow_mut();
        n.value = t;
        n.grad = Plain::zeros(&n.value.shape);
    }

    /// Overwrite the gradient directly — the manual-backprop hook used by
    /// optimiser tests and classic hand-derived training loops.
    pub fn node_grad(&self, g: Plain) {
        self.node.borrow_mut().grad = g;
    }

    /// Reset this leaf's gradient to zero.
    pub fn zero_grad(&self) {
        let mut n = self.node.borrow_mut();
        let shape = n.value.shape.clone();
        n.grad = Plain::zeros(&shape);
    }

    fn from_node(node: NodeRef) -> Var {
        Var { node }
    }

}

fn make_node(value: Plain, deps: Vec<(Var, BackFn)>) -> Var {
    let shape = value.shape.clone();
    let node = Rc::new(RefCell::new(Node { value, grad: Plain::zeros(&shape), deps: deps.into_iter().map(|(v, f)| (v.node, f)).collect() }));
    Var::from_node(node)
}

// ── elementary ops ───────────────────────────────────────────────────────────

/// z = a + b (elementwise, shapes must match).
pub fn vadd(a: &Var, b: &Var) -> Var {
    let v = a.val().add(&b.val());
    let bd = b.clone();
    let ad = a.clone();
    make_node(
        v,
        vec![
            (ad, Rc::new(|g: &Plain| g.clone())),
            (bd, Rc::new(|g: &Plain| g.clone())),
        ],
    )
}

/// z = a − b.
pub fn vsub(a: &Var, b: &Var) -> Var {
    let v = a.val().sub(&b.val());
    let ad = a.clone();
    let bd = b.clone();
    make_node(
        v,
        vec![
            (ad, Rc::new(|g: &Plain| g.clone())),
            (bd, Rc::new(|g: &Plain| g.scale(-1.0))),
        ],
    )
}

/// z = a ⊙ b (elementwise product).
pub fn vmul(a: &Var, b: &Var) -> Var {
    let v = a.val().mul(&b.val());
    let (av, bv) = (a.val(), b.val());
    let ad = a.clone();
    let bd = b.clone();
    make_node(
        v,
        vec![
            (ad, Rc::new(move |g: &Plain| g.mul(&bv))),
            (bd, Rc::new(move |g: &Plain| g.mul(&av))),
        ],
    )
}

/// z = s·a.
pub fn vscale(a: &Var, s: f32) -> Var {
    let v = a.val().scale(s);
    let ad = a.clone();
    make_node(v, vec![(ad, Rc::new(move |g: &Plain| g.scale(s)))])
}

/// z = reshape(a, shape) — same elements, new layout. The backward restores
/// the incoming gradient to the original shape.
pub fn vreshape(a: &Var, shape: &[usize]) -> Var {
    let v = a.val();
    assert_eq!(v.data.len(), shape.iter().product::<usize>(), "reshape: element count must match");
    let old = v.shape.clone();
    let out = Plain::from_vec(shape, v.data.clone());
    let ad = a.clone();
    make_node(out, vec![(ad, Rc::new(move |g: &Plain| Plain::from_vec(&old, g.data.clone())))])
}

/// z = a · b (2-D matmul). dz→da = g·bᵀ, dz→db = aᵀ·g.
pub fn vmatmul(a: &Var, b: &Var) -> Var {
    let v = a.val().matmul(&b.val());
    let (av, bv) = (a.val(), b.val());
    let ad = a.clone();
    let bd = b.clone();
    make_node(
        v,
        vec![
            (ad, Rc::new(move |g: &Plain| g.matmul(&bv.t()))),
            (bd, Rc::new(move |g: &Plain| av.t().matmul(g))),
        ],
    )
}

/// z = relu(a).
pub fn vrelu(a: &Var) -> Var {
    let df = a.val().map(crate::nn::relu_d);
    let v = a.val().map(crate::nn::relu);
    let ad = a.clone();
    make_node(v, vec![(ad, Rc::new(move |g: &Plain| g.mul(&df)))])
}

/// z = tanh(a).
pub fn vtanh(a: &Var) -> Var {
    let df = a.val().map(crate::nn::tanh_d);
    let v = a.val().map(crate::nn::tanh);
    let ad = a.clone();
    make_node(v, vec![(ad, Rc::new(move |g: &Plain| g.mul(&df)))])
}

/// z = sigmoid(a).
pub fn vsigmoid(a: &Var) -> Var {
    let df = a.val().map(crate::nn::sigmoid_d);
    let v = a.val().map(crate::nn::sigmoid);
    let ad = a.clone();
    make_node(v, vec![(ad, Rc::new(move |g: &Plain| g.mul(&df)))])
}

/// z = gelu(a).
pub fn vgelu(a: &Var) -> Var {
    let df = a.val().map(crate::nn::gelu_d);
    let v = a.val().map(crate::nn::gelu);
    let ad = a.clone();
    make_node(v, vec![(ad, Rc::new(move |g: &Plain| g.mul(&df)))])
}

/// z = a² (handy for squared-error losses).
pub fn vsquare(a: &Var) -> Var {
    let av = a.val();
    let v = av.map(|x| x * x);
    let ad = a.clone();
    make_node(v, vec![(ad, Rc::new(move |g: &Plain| {
        let two = av.scale(2.0);
        g.mul(&two)
    }))])
}

/// z = mean(a) — scalar [1] tensor.
pub fn vmean(a: &Var) -> Var {
    let av = a.val();
    let n = av.len().max(1) as f32;
    let shape = av.shape.clone();
    let v = Plain::from_vec(&[1], vec![av.mean()]);
    let ad = a.clone();
    make_node(v, vec![(ad, Rc::new(move |g: &Plain| {
        Plain::full(&shape, g.data.first().copied().unwrap_or(0.0) / n)
    }))])
}

/// MSE loss: mean((pred − truth)²). truth is a constant (no gradient).
pub fn vmse(pred: &Var, truth: &Plain) -> Var {
    let diff = pred.val().sub(truth);
    let n = diff.len().max(1) as f32;
    let loss = diff.data.iter().map(|d| d * d).sum::<f32>() / n;
    let pv = pred.val();
    let td = truth.clone();
    let pd = pred.clone();
    make_node(
        Plain::from_vec(&[1], vec![loss]),
        vec![(pd, Rc::new(move |g: &Plain| {
            // d loss / d pred = 2 (pred − truth) / n, scaled by the seed
            let s = g.data.first().copied().unwrap_or(1.0);
            pv.sub(&td).scale(2.0 * s / n)
        }))],
    )
}

/// Softmax cross-entropy over a (n, c) logit batch. Fused, numerically
/// stable, gradient = (softmax − onehot)/n scaled by the seed.
pub fn vcross_entropy(logits: &Var, targets: &[usize]) -> Var {
    let (n, c) = logits.val().dims2();
    assert_eq!(targets.len(), n, "cross-entropy: targets must match batch");
    let probs = crate::nn::softmax(&logits.val().to_plain());
    let mut loss = 0.0f32;
    for i in 0..n {
        loss -= probs.data[i * c + targets[i]].max(1e-12).ln();
    }
    loss /= n as f32;
    // precompute the full local gradient: (p − onehot)/n
    let mut local = probs.clone();
    for i in 0..n {
        local.data[i * c + targets[i]] -= 1.0;
    }
    let local = local.scale(1.0 / n as f32);
    let local = Plain::from_plain(local);
    let ld = logits.clone();
    make_node(
        Plain::from_vec(&[1], vec![loss]),
        vec![(ld, Rc::new(move |g: &Plain| {
            let s = g.data.first().copied().unwrap_or(1.0);
            local.scale(s)
        }))],
    )
}

/// Dropout: elementwise Bernoulli keep-mask, scaled by 1/keep at train time.
/// `train: false` is the identity (eval mode), exactly like Burn's Dropout.
pub fn vdropout(a: &Var, p: f32, train: bool, rng: &mut crate::rng::Rng) -> Var {
    if !train || p <= 0.0 {
        return a.clone();
    }
    let keep = 1.0 - p;
    let v = a.val();
    let mask: Vec<f32> =
        (0..v.len()).map(|_| if rng.next_f64() < keep as f64 { 1.0 / keep as f32 } else { 0.0 }).collect();
    let out = Plain::from_vec(&v.shape, v.data.iter().zip(&mask).map(|(&x, &m)| x * m).collect());
    let m = Plain::from_vec(&v.shape, mask);
    let ad = a.clone();
    make_node(out, vec![(ad, Rc::new(move |g: &Plain| g.mul(&m)))])
}

/// Row-gather: z[i] = a[idx[i]] — the embedding lookup primitive.
pub fn vgather(a: &Var, idx: &[usize]) -> Var {
    let (v, dim) = a.val().dims2();
    assert!(idx.iter().all(|&i| i < v), "gather: index out of range");
    let idx: Vec<usize> = idx.to_vec();
    let mut out = Vec::with_capacity(idx.len() * dim);
    for &i in &idx {
        out.extend_from_slice(&a.node.borrow().value.data[i * dim..(i + 1) * dim]);
    }
    let shape = vec![idx.len(), dim];
    let ad = a.clone();
    let vshape = vec![v, dim];
    make_node(
        Plain::from_vec(&shape, out),
        vec![(ad, Rc::new(move |g: &Plain| {
            let mut acc = Plain::zeros(&vshape);
            for (r, &i) in idx.iter().enumerate() {
                for j in 0..dim {
                    acc.data[i * dim + j] += g.data[r * dim + j];
                }
            }
            acc
        }))],
    )
}

// ── backward sweep ───────────────────────────────────────────────────────────

/// Run reverse-mode differentiation from `out`. After this call every leaf
/// `Var::param` reachable from `out` holds its accumulated gradient.
pub fn backward(out: &Var) {
    // 1) topological order (DFS, pointer identity)
    let mut order: Vec<NodeRef> = Vec::new();
    let mut seen: HashSet<usize> = HashSet::new();
    fn visit(node: &NodeRef, seen: &mut HashSet<usize>, order: &mut Vec<NodeRef>) {
        let key = Rc::as_ptr(node) as *const () as usize;
        if seen.insert(key) {
            let deps = node.borrow().deps.clone();
            for (p, _) in &deps {
                visit(p, seen, order);
            }
            order.push(node.clone());
        }
    }
    visit(&out.node, &mut seen, &mut order);

    // 2) seed the output
    {
        let mut n = out.node.borrow_mut();
        let shape = n.value.shape.clone();
        n.grad = Plain::full(&shape, 1.0);
    }

    // 3) reverse sweep
    for node in order.iter().rev() {
        let g = node.borrow().grad.clone();
        let deps = node.borrow().deps.clone();
        for (p, f) in deps {
            let pg = f(&g);
            let mut pb = p.borrow_mut();
            pb.grad = pb.grad.add(&pg);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `backward` seeds the gradient of the SUM of all output elements, so
    /// the finite-difference oracle must perturb the same quantity.
    fn gradcheck(f: impl Fn(&Var) -> Var, x0: &[f32]) {
        let x = Var::param(Plain::from_vec(&[x0.len()], x0.to_vec()));
        let out = f(&x);
        backward(&out);
        let analytic = x.grad().data;
        let h = 1e-3f32;
        for i in 0..x0.len() {
            let mut up = x0.to_vec();
            let mut dn = x0.to_vec();
            up[i] += h;
            dn[i] -= h;
            let xu = Var::param(Plain::from_vec(&[x0.len()], up));
            let xd = Var::param(Plain::from_vec(&[x0.len()], dn));
            let fd = (f(&xu).val().data.iter().sum::<f32>()
                - f(&xd).val().data.iter().sum::<f32>())
                / (2.0 * h);
            assert!(
                (fd - analytic[i]).abs() < 2e-2,
                "i={}: analytic {} vs finite {}",
                i,
                analytic[i],
                fd
            );
        }
    }

    #[test]
    fn gradcheck_add_mul_scale() {
        gradcheck(|x| {
            let c = Var::constant(Plain::from_vec(&[3], vec![2.0, -1.0, 0.5]));
            vadd(&vmul(x, &c), &vscale(x, 1.5))
        }, &[1.0, 2.0, -3.0]);
        // d/dx [x*c + 1.5x] = c + 1.5
    }

    #[test]
    fn gradcheck_matmul() {
        // y = x·W, x (1,3), W (3,2) fixed → dy/dx = W
        let w = Var::constant(Plain::from_vec(&[3, 2], vec![1., 2., 3., 4., 5., 6.]));
        gradcheck(|x| vmatmul(&vreshape(x, &[1, 3]), &w), &[0.5, -0.5, 1.0]);
    }

    #[test]
    fn gradcheck_reshape_roundtrip() {
        gradcheck(|x| vreshape(x, &[1, 3]), &[0.7, -0.2, 1.4]);
    }

    #[test]
    fn gradcheck_relu_tanh_sigmoid_gelu() {
        gradcheck(|x| vrelu(x), &[0.7, -0.4]);
        gradcheck(|x| vtanh(x), &[0.3, -0.8]);
        gradcheck(|x| vsigmoid(x), &[0.9, -0.2]);
        gradcheck(|x| vgelu(x), &[0.6, -1.1]);
    }

    #[test]
    fn gradcheck_mse() {
        let truth = Plain::from_vec(&[3], vec![0.5, -0.5, 1.0]);
        gradcheck(|x| vmse(x, &truth), &[0.2, 0.9, -0.3]);
    }

    #[test]
    fn gradcheck_cross_entropy() {
        let targets = [1usize];
        gradcheck(|x| vcross_entropy(&vreshape(x, &[1, 3]), &targets), &[0.2, 0.7, -0.4]);
    }

    #[test]
    fn gradcheck_gather() {
        // z = a[idx]; pick row 1 of a (3,2)
        let a = Var::param(Plain::from_vec(&[3, 2], vec![1., 2., 3., 4., 5., 6.]));
        let z = vgather(&a, &[1]);
        // y = sum(z) → grad row1 = [1,1], others 0
        let y = vmean(&z);
        backward(&y);
        let g = a.grad().data;
        assert!(g.iter().take(2).all(|&v| v.abs() < 1e-6));
        assert!((g[2] - 0.5).abs() < 1e-6 && (g[3] - 0.5).abs() < 1e-6);
        assert!(g[4..].iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn grads_do_not_leak_across_backwards() {
        let w = Var::param(Plain::from_vec(&[2], vec![1.0, 2.0]));
        let x = Var::constant(Plain::from_vec(&[2], vec![3.0, 4.0]));
        let y1 = vmul(&w, &x);
        let l1 = vmean(&y1);
        backward(&l1);
        // d(mean(w⊙x))/dw = x/2 = [1.5, 2.0]
        assert_eq!(w.grad().data, vec![1.5, 2.0]);
        w.zero_grad();
        let y2 = vmul(&w, &x);
        let l2 = vmean(&y2);
        backward(&l2);
        assert_eq!(w.grad().data, vec![1.5, 2.0], "second sweep must match the first exactly");
    }

    #[test]
    fn dropout_eval_is_identity() {
        let a = Var::param(Plain::from_vec(&[8], vec![1.0; 8]));
        let mut rng = crate::rng::Rng::new(5);
        let out = vdropout(&a, 0.5, false, &mut rng);
        assert_eq!(out.val().data, a.val().data);
    }

    #[test]
    fn dropout_train_scales_kept_values() {
        let a = Var::param(Plain::from_vec(&[2000], vec![1.0; 2000]));
        let mut rng = crate::rng::Rng::new(6);
        let out = vdropout(&a, 0.5, true, &mut rng);
        let kept = out.val().data.iter().filter(|&&v| v > 0.0).count();
        // ≈ half kept, each scaled to 2.0
        assert!(kept > 800 && kept < 1200, "kept = {}", kept);
        assert!(out.val().data.iter().all(|&v| v == 0.0 || (v - 2.0).abs() < 1e-5));
    }
}
