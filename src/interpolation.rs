//! Polynomial interpolation: the lowest-degree polynomial through a set of points.

use crate::polynomials::{Polynomial, PolynomialRing};
use crate::structures::{CommutativeMonoid, Domain, Field, RingOps};
use std::rc::Rc;

/// Multiplies every coefficient by `c`.
///
/// Both clones are forced. `p` is a basis polynomial that later calls reuse, so its
/// coefficients cannot be moved out, and the owned `Mul` consumes `c`. The borrowed
/// form would need `for<'a> &'a R::E: Mul<&'a R::E>`, which overflows trait
/// resolution through `Matrix`'s own recursive borrowed operators.
fn scale<R>(ring: &Rc<PolynomialRing<R>>, p: &Polynomial<R>, c: &R::E) -> Polynomial<R>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    ring.element(
        p.coefficients()
            .iter()
            .map(|a| a.clone() * c.clone())
            .collect::<Vec<_>>(),
    )
}

/// Interpolation at a fixed set of nodes.
///
/// The Lagrange basis depends only on the x values, so building it once makes each
/// later interpolation a linear combination: `O(k^2)` to construct, `O(k^2)` per call
/// for the scaling, against `O(k^2)` polynomial products if rebuilt each time.
pub struct Interpolation<R>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    ring: Rc<PolynomialRing<R>>,
    /// `basis[j]` is 1 at `x[j]` and 0 at every other node.
    basis: Vec<Polynomial<R>>,
}

impl<R> Interpolation<R>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    /// Builds the Lagrange basis for `nodes`, or [`None`] if two of them coincide.
    pub fn new(ring: &Rc<PolynomialRing<R>>, nodes: &[R::E]) -> Option<Self> {
        let field = ring.coefficients().clone();
        let basis = (0..nodes.len())
            .map(|j| {
                // l_j = prod_{m != j} (X - x_m) / (x_j - x_m). Gathering the scalars
                // needs one inversion, and starting the product from that constant
                // folds the scaling in rather than making a second pass.
                let mut denominator = field.identity();
                for (m, x_m) in nodes.iter().enumerate() {
                    if m != j {
                        denominator *= nodes[j].clone() - x_m.clone();
                    }
                }
                let mut l = ring.element(vec![field.invert(&denominator)?]);
                for (m, x_m) in nodes.iter().enumerate() {
                    if m != j {
                        l *= ring.element(vec![field.zero() - x_m.clone(), field.identity()]);
                    }
                }
                Some(l)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Interpolation {
            ring: Rc::clone(ring),
            basis,
        })
    }

    /// How many nodes this was built for.
    pub fn nodes(&self) -> usize {
        self.basis.len()
    }

    /// The polynomial taking `values[j]` at node `j`.
    ///
    /// # Panics
    /// If `values` does not have one entry per node.
    pub fn apply(&self, values: &[R::E]) -> Polynomial<R> {
        assert_eq!(
            values.len(),
            self.basis.len(),
            "expected one value per node"
        );
        self.basis
            .iter()
            .zip(values)
            .fold(self.ring.zero(), |sum, (l, y)| {
                sum + scale(&self.ring, l, y)
            })
    }
}

/// The lowest-degree polynomial with `p(x[i]) == y[i]`, or [`None`] if two x values
/// coincide.
///
/// Builds the Lagrange basis and discards it. Use [`Interpolation`] to interpolate
/// repeatedly at the same nodes.
///
/// # Panics
/// If `x` and `y` differ in length.
pub fn interpolate<R>(
    ring: &Rc<PolynomialRing<R>>,
    x: &[R::E],
    y: &[R::E],
) -> Option<Polynomial<R>>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    assert_eq!(x.len(), y.len(), "x and y must have the same length");
    Some(Interpolation::new(ring, x)?.apply(y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::Integers;
    use crate::polynomials::RingExt;
    use crate::structures::QuotientRing;

    /// F_101, with room for several distinct nodes.
    fn field() -> Rc<QuotientRing<Integers>> {
        Integers::modulo(101)
    }

    #[test]
    fn recovers_a_polynomial_from_its_values() {
        let f = field();
        let ring = f.polynomials();

        // p = 3 + 2x + 5x^3, sampled at five nodes.
        let p = ring.element(vec![f.element(3), f.element(2), f.element(0), f.element(5)]);
        let x: Vec<_> = (1..=5i64).map(|n| f.element(n)).collect();
        let y: Vec<_> = x.iter().map(|xi| p.evaluate(xi)).collect();

        let q = interpolate(&ring, &x, &y).expect("nodes are distinct");
        assert_eq!(q, p);
        // Degree is at most one less than the node count, and here matches p exactly.
        assert_eq!(q.degree(), Some(3));
    }

    #[test]
    fn passes_through_every_point() {
        let f = field();
        let ring = f.polynomials();
        let x: Vec<_> = [0i64, 2, 7, 13].iter().map(|n| f.element(*n)).collect();
        let y: Vec<_> = [5i64, 9, 1, 40].iter().map(|n| f.element(*n)).collect();

        let p = interpolate(&ring, &x, &y).unwrap();
        for (xi, yi) in x.iter().zip(&y) {
            assert_eq!(&p.evaluate(xi), yi);
        }
        // Four nodes, so degree at most three.
        assert!(p.degree().unwrap() <= 3);

        // A single point gives the constant.
        let c = interpolate(&ring, &x[..1], &y[..1]).unwrap();
        assert_eq!(c, ring.element(vec![f.element(5)]));
        assert_eq!(c.degree(), Some(0));
    }

    #[test]
    fn repeated_nodes_have_no_interpolant() {
        let f = field();
        let ring = f.polynomials();
        let x = vec![f.element(4), f.element(9), f.element(4)];
        let y = vec![f.element(1), f.element(2), f.element(3)];
        assert!(interpolate(&ring, &x, &y).is_none());
        assert!(Interpolation::new(&ring, &x).is_none());
    }

    #[test]
    fn the_basis_is_reusable_across_value_sets() {
        let f = field();
        let ring = f.polynomials();
        let x: Vec<_> = (1..=4i64).map(|n| f.element(n)).collect();
        let interpolation = Interpolation::new(&ring, &x).unwrap();
        assert_eq!(interpolation.nodes(), 4);

        for values in [vec![1i64, 0, 0, 0], vec![0, 1, 0, 0], vec![7, 7, 7, 7]] {
            let y: Vec<_> = values.iter().map(|n| f.element(*n)).collect();
            let p = interpolation.apply(&y);
            for (xi, yi) in x.iter().zip(&y) {
                assert_eq!(&p.evaluate(xi), yi);
            }
            // Same answer as the one-shot entry point.
            assert_eq!(p, interpolate(&ring, &x, &y).unwrap());
        }

        // A constant set of values interpolates to that constant, not a higher-degree
        // polynomial that happens to agree.
        let sevens: Vec<_> = (0..4).map(|_| f.element(7)).collect();
        assert_eq!(interpolation.apply(&sevens).degree(), Some(0));
    }

    #[test]
    #[should_panic(expected = "one value per node")]
    fn rejects_a_mismatched_value_count() {
        let f = field();
        let ring = f.polynomials();
        let x: Vec<_> = (1..=3i64).map(|n| f.element(n)).collect();
        Interpolation::new(&ring, &x).unwrap().apply(&[f.element(1)]);
    }
}
