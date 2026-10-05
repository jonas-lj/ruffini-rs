//! Polynomial interpolation: the lowest-degree polynomial through a set of points.

use crate::polynomials::{Polynomial, PolynomialRing};
use crate::structures::{CommutativeMonoid, Domain, Field, RingOps};
use std::sync::Arc;

/// Multiplies every coefficient by `c`.
// Both clones are forced: the basis is reused, so its coefficients cannot move out,
// and the borrowed Mul needs an HRTB that overflows through Matrix.
fn scale<R>(ring: &Arc<PolynomialRing<R>>, p: &Polynomial<R>, c: &R::E) -> Polynomial<R>
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

/// Interpolation at a fixed set of x values, keeping the Lagrange basis for reuse.
// The basis depends only on the x values, so each later call is a linear combination
// rather than k^2 further polynomial products.
pub struct Interpolation<R>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    ring: Arc<PolynomialRing<R>>,
    /// `basis[j]` is 1 at `xs[j]` and 0 at every other x value.
    basis: Vec<Polynomial<R>>,
}

impl<R> Interpolation<R>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    /// Builds the Lagrange basis for `xs`, or [`None`] if two of them coincide.
    pub fn new(ring: &Arc<PolynomialRing<R>>, xs: &[R::E]) -> Option<Self> {
        let field = ring.coefficients().clone();
        let x = ring.indeterminate();
        let lifted: Vec<Polynomial<R>> = xs.iter().map(|x_m| ring.constant(x_m.clone())).collect();

        let basis = (0..xs.len())
            .map(|j| {
                // l_j = prod_{m != j} (x - x_m) / (x_j - x_m). Starting the product from
                // the scalar folds the division in, avoiding a second pass.
                let mut denominator = field.identity();
                for (m, x_m) in xs.iter().enumerate() {
                    if m != j {
                        denominator *= xs[j].clone() - x_m.clone();
                    }
                }

                let mut l = ring.constant(field.invert(&denominator)?);
                for (m, x_m) in lifted.iter().enumerate() {
                    if m != j {
                        l *= x.clone() - x_m.clone();
                    }
                }
                Some(l)
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Interpolation {
            ring: Arc::clone(ring),
            basis,
        })
    }

    /// How many x values this was built for, which bounds the interpolant's degree.
    pub fn len(&self) -> usize {
        self.basis.len()
    }

    /// Whether this was built for no x values at all.
    pub fn is_empty(&self) -> bool {
        self.basis.is_empty()
    }

    /// The polynomial taking `values[j]` at `xs[j]`. Panics on a wrong value count.
    pub fn apply(&self, values: &[R::E]) -> Polynomial<R> {
        assert_eq!(values.len(), self.basis.len(), "expected one value per x");
        self.basis
            .iter()
            .zip(values)
            .fold(self.ring.zero(), |sum, (l, y)| {
                sum + scale(&self.ring, l, y)
            })
    }
}

/// The lowest-degree polynomial with `p(xs[i]) == ys[i]`, or [`None`] if two x values
/// coincide. See [`Interpolation`] to reuse the basis; panics on mismatched lengths.
pub fn interpolate<R>(
    ring: &Arc<PolynomialRing<R>>,
    xs: &[R::E],
    ys: &[R::E],
) -> Option<Polynomial<R>>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    assert_eq!(xs.len(), ys.len(), "xs and ys must have the same length");
    Some(Interpolation::new(ring, xs)?.apply(ys))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::Integers;
    use crate::polynomials::RingExt;
    use crate::structures::QuotientRing;

    /// F_101, with room for several distinct x values.
    fn field() -> Arc<QuotientRing<Integers>> {
        Integers::modulo(101)
    }

    #[test]
    fn recovers_a_polynomial_from_its_values() {
        let f = field();
        let ring = f.polynomials();

        // p = 3 + 2x + 5x^3, sampled at five x values.
        let p = ring.element(vec![f.element(3), f.element(2), f.element(0), f.element(5)]);
        let x: Vec<_> = (1..=5i64).map(|n| f.element(n)).collect();
        let y: Vec<_> = x.iter().map(|xi| p.evaluate(xi)).collect();

        let q = interpolate(&ring, &x, &y).expect("the x values are distinct");
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
        // Four x values, so degree at most three.
        assert!(p.degree().unwrap() <= 3);

        // A single point gives the constant.
        let c = interpolate(&ring, &x[..1], &y[..1]).unwrap();
        assert_eq!(c, ring.element(vec![f.element(5)]));
        assert_eq!(c.degree(), Some(0));
    }

    #[test]
    fn a_repeated_x_value_has_no_interpolant() {
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
        assert_eq!(interpolation.len(), 4);

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
    #[should_panic(expected = "one value per x")]
    fn rejects_a_mismatched_value_count() {
        let f = field();
        let ring = f.polynomials();
        let x: Vec<_> = (1..=3i64).map(|n| f.element(n)).collect();
        Interpolation::new(&ring, &x).unwrap().apply(&[f.element(1)]);
    }
}
