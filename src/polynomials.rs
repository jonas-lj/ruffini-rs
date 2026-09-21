//! Univariate polynomials with coefficients in an arbitrary [`Ring`].
//!
//! `R[x]` is itself a ring, so this module provides the [`PolynomialRing`]
//! value-domain handle and the [`Polynomial`] element type, mirroring the
//! [`crate::structures::QuotientRing`] pattern.

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, DivRem, EuclideanDomain, Field, Monoid, Ring, RingOps,
    SemiRing, Semigroup, Domain,
};
use itertools::{EitherOrBoth, Itertools};
use num_bigint::BigInt;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use std::rc::Rc;

/// The polynomial ring `R[x]` over a coefficient ring `R`.
#[derive(Debug, Clone)]
pub struct PolynomialRing<R: Ring>
where
    R::E: RingOps + Eq,
{
    coeff_ring: R,
}

/// A polynomial with coefficients in `R`, stored in increasing-degree order
/// (constant term first).
///
/// Invariant: the coefficient vector has no trailing zeros — the zero polynomial
/// is the empty vector. Equality is structural over the coefficient vector.
#[derive(Debug)]
pub struct Polynomial<R: Ring>
where
    R::E: RingOps + Eq,
{
    coefficients: Vec<R::E>,
    ring: Rc<PolynomialRing<R>>,
}

/// Puts [`polynomials`](RingExt::polynomials) on every ring, so `structures` does not
/// have to name this module.
pub trait RingExt: Ring + Clone + Sized
where
    Self::E: RingOps + Eq,
{
    /// The polynomial ring `Self[x]`.
    fn polynomials(&self) -> Rc<PolynomialRing<Self>> {
        PolynomialRing::new(self.clone())
    }
}

impl<R> RingExt for R
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
}

impl<R> PolynomialRing<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    /// Construct `R[x]`, wrapped in an [`Rc`] so polynomials can share the ring handle.
    pub fn new(coeff_ring: R) -> Rc<Self> {
        Rc::new(PolynomialRing { coeff_ring })
    }

    /// The underlying coefficient ring.
    pub fn coefficients(&self) -> &R {
        &self.coeff_ring
    }
}

impl<R> Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    /// Degree of the polynomial, or `None` for the zero polynomial.
    pub fn degree(&self) -> Option<usize> {
        if self.coefficients.is_empty() {
            None
        } else {
            Some(self.coefficients.len() - 1)
        }
    }

    /// Coefficients in increasing-degree order.
    pub fn coefficients(&self) -> &[R::E] {
        &self.coefficients
    }

    /// Evaluates at `x` by Horner's method, in `degree` multiplications.
    pub fn evaluate(&self, x: &R::E) -> R::E {
        self.coefficients
            .iter()
            .rev()
            .fold(self.ring.coeff_ring.zero(), |acc, c| {
                acc * x.clone() + c.clone()
            })
    }

    /// The evaluation map as a closure, for where a function value is wanted.
    // Droppable once fn_traits is stable: Polynomial could implement Fn directly.
    pub fn as_fn(&self) -> impl Fn(&R::E) -> R::E + '_ {
        move |x| self.evaluate(x)
    }
}

impl<R> Clone for Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn clone(&self) -> Self {
        Polynomial {
            coefficients: self.coefficients.clone(),
            ring: Rc::clone(&self.ring),
        }
    }
}

impl<R> PartialEq for Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &other.ring),
            "Polynomial operands belong to different rings"
        );
        self.coefficients == other.coefficients
    }
}

impl<R> Eq for Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Add for Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        let result: Vec<R::E> = self
            .coefficients
            .into_iter()
            .zip_longest(rhs.coefficients)
            .map(|pair| match pair {
                EitherOrBoth::Both(x, y) => x + y,
                EitherOrBoth::Left(x) | EitherOrBoth::Right(x) => x,
            })
            .collect();
        self.ring.element(result)
    }
}

impl<R> Sub for Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        let zero = self.ring.coeff_ring.zero();
        let result: Vec<R::E> = self
            .coefficients
            .into_iter()
            .zip_longest(rhs.coefficients)
            .map(|pair| match pair {
                EitherOrBoth::Both(x, y) => x - y,
                EitherOrBoth::Left(x) => x,
                // Owned Sub cannot take a borrowed-operand bound without RingOps
                // dragging it through nested polynomials, so zero is replaced per term.
                EitherOrBoth::Right(y) => zero.clone() - y,
            })
            .collect();
        self.ring.element(result)
    }
}

impl<R> Mul for Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        if self.coefficients.is_empty() || rhs.coefficients.is_empty() {
            return Polynomial {
                coefficients: Vec::new(),
                ring: self.ring,
            };
        }
        let (a, b) = (&self.coefficients, &rhs.coefficients);
        let (n, m) = (a.len(), b.len());
        let zero = self.ring.coeff_ring.zero();
        // Each output coefficient is the sum over i + j == k, gathered independently of
        // the others, so the outer map has no shared state to contend over.
        let result: Vec<R::E> = (0..n + m - 1)
            .map(|k| {
                (k.saturating_sub(m - 1)..=k.min(n - 1))
                    .map(|i| a[i].clone() * b[k - i].clone())
                    .fold(zero.clone(), |sum, term| sum + term)
            })
            .collect();
        self.ring.element(result)
    }
}

/// Coefficient-wise addition of two borrowed polynomials; the shorter one is padded
/// by carrying its counterpart's remaining coefficients through unchanged.
impl<R> Add<&Polynomial<R>> for &Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
    for<'c> &'c R::E: Add<&'c R::E, Output = R::E>,
{
    type Output = Polynomial<R>;
    fn add(self, rhs: &Polynomial<R>) -> Self::Output {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        let result: Vec<R::E> = self
            .coefficients
            .iter()
            .zip_longest(&rhs.coefficients)
            .map(|pair| match pair {
                EitherOrBoth::Both(x, y) => x + y,
                EitherOrBoth::Left(x) | EitherOrBoth::Right(x) => x.clone(),
            })
            .collect();
        self.ring.element(result)
    }
}

/// As [`Add`], except a coefficient present only in `rhs` is negated as `0 - y`.
impl<R> Sub<&Polynomial<R>> for &Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
    for<'c> &'c R::E: Sub<&'c R::E, Output = R::E>,
{
    type Output = Polynomial<R>;
    fn sub(self, rhs: &Polynomial<R>) -> Self::Output {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        let zero = self.ring.coeff_ring.zero();
        let result: Vec<R::E> = self
            .coefficients
            .iter()
            .zip_longest(&rhs.coefficients)
            .map(|pair| match pair {
                EitherOrBoth::Both(x, y) => x - y,
                EitherOrBoth::Left(x) => x.clone(),
                EitherOrBoth::Right(y) => &zero - y,
            })
            .collect();
        self.ring.element(result)
    }
}

/// Schoolbook convolution of the coefficient vectors.
impl<R> Mul<&Polynomial<R>> for &Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
    for<'c> &'c R::E: Mul<&'c R::E, Output = R::E>,
{
    type Output = Polynomial<R>;
    fn mul(self, rhs: &Polynomial<R>) -> Self::Output {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        if self.coefficients.is_empty() || rhs.coefficients.is_empty() {
            return Polynomial {
                coefficients: Vec::new(),
                ring: Rc::clone(&self.ring),
            };
        }
        let (a, b) = (&self.coefficients, &rhs.coefficients);
        let (n, m) = (a.len(), b.len());
        let zero = self.ring.coeff_ring.zero();
        let result: Vec<R::E> = (0..n + m - 1)
            .map(|k| {
                (k.saturating_sub(m - 1)..=k.min(n - 1))
                    .map(|i| &a[i] * &b[k - i])
                    .fold(zero.clone(), |sum, term| sum + term)
            })
            .collect();
        self.ring.element(result)
    }
}

forward_ref_binops!(
    Polynomial<R>,
    { R: Ring, R::E: RingOps + Eq, },
    Add, add; Sub, sub; Mul, mul
);

int_operand_ops!(
    Polynomial<R>,
    { R: Ring, R::E: RingOps + Eq, },
    i64,
    BigInt
);

impl<R> fmt::Display for Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.coefficients.is_empty() {
            return write!(f, "0");
        }
        for (i, c) in self.coefficients.iter().enumerate() {
            if i > 0 {
                write!(f, " + ")?;
            }
            // A coefficient that is itself a sum has to be bracketed, or a nested
            // polynomial's terms run together with the outer ones.
            let c = c.to_string();
            let c = if c.contains(' ') {
                format!("({})", c)
            } else {
                c
            };
            match i {
                0 => write!(f, "{}", c)?,
                1 => write!(f, "{}*x", c)?,
                k => write!(f, "{}*x^{{{}}}", c, k)?,
            }
        }
        Ok(())
    }
}

/// Compound assignment.
///
/// These go through the owned operators rather than the borrowed ones. Requiring
/// `&Polynomial<R>: Add<&Polynomial<R>>` here sends trait resolution around a cycle,
/// because `R::E` may itself be a `Polynomial`. Swapping in the zero polynomial costs
/// only an `Rc` bump, so the left operand is still not cloned.
macro_rules! polynomial_assign_ops {
    ($($op:ident, $method:ident, $base_method:ident);* $(;)?) => {$(
        impl<R> $op<&Polynomial<R>> for Polynomial<R>
        where
            R: Ring,
            R::E: RingOps + Eq,
        {
            fn $method(&mut self, rhs: &Polynomial<R>) {
                let placeholder = Polynomial {
                    coefficients: Vec::new(),
                    ring: Rc::clone(&self.ring),
                };
                let lhs = std::mem::replace(self, placeholder);
                *self = lhs.$base_method(rhs.clone());
            }
        }

        impl<R> $op<Polynomial<R>> for Polynomial<R>
        where
            R: Ring,
            R::E: RingOps + Eq,
        {
            fn $method(&mut self, rhs: Polynomial<R>) {
                let placeholder = Polynomial {
                    coefficients: Vec::new(),
                    ring: Rc::clone(&self.ring),
                };
                let lhs = std::mem::replace(self, placeholder);
                *self = lhs.$base_method(rhs);
            }
        }
    )*};
}
polynomial_assign_ops!(AddAssign, add_assign, add; MulAssign, mul_assign, mul);


impl<R> Domain for Rc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type E = Polynomial<R>;
    type Repr = Vec<R::E>;

    /// Trims trailing zero coefficients to keep the canonical form.
    fn element<T: Into<Vec<R::E>>>(&self, coefficients: T) -> Self::E {
        let mut coefficients = coefficients.into();
        let zero = self.coeff_ring.zero();
        while coefficients.last() == Some(&zero) {
            coefficients.pop();
        }
        Polynomial {
            coefficients,
            ring: Rc::clone(self),
        }
    }
}

impl<R> Semigroup for Rc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Monoid for Rc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn identity(&self) -> Self::E {
        self.element(vec![self.coeff_ring.identity()])
    }
}

impl<R> CommutativeMonoid for Rc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn zero(&self) -> Self::E {
        self.element(Vec::new())
    }
}

impl<R> AdditiveGroup for Rc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> SemiRing for Rc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Ring for Rc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

/// Polynomial long division. Requires `R: Field` so leading coefficients can be inverted.
/// Panics if the divisor is the zero polynomial.
impl<R> DivRem for Polynomial<R>
where
    R: Field,
    R::E: RingOps + Eq,
{
    fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &divisor.ring),
            "Polynomial operands belong to different rings"
        );
        let coeff_ring = &self.ring.coeff_ring;
        let zero_c = coeff_ring.zero();

        let lead_b = divisor
            .coefficients
            .last()
            .expect("division by zero polynomial");
        let lead_b_inv = coeff_ring
            .invert(lead_b)
            .expect("leading coefficient must be invertible in a field");
        let deg_b = divisor.coefficients.len() - 1;

        let mut r: Vec<R::E> = self.coefficients.clone();
        let mut q: Vec<R::E> = Vec::new();

        while r.len() > deg_b {
            // c = leading(r) / leading(divisor)
            let c = r.last().unwrap().clone() * lead_b_inv.clone();
            let k = r.len() - 1 - deg_b;

            // r -= c * x^k * divisor
            for j in 0..divisor.coefficients.len() {
                let term = c.clone() * divisor.coefficients[j].clone();
                let prev = std::mem::replace(&mut r[k + j], zero_c.clone());
                r[k + j] = prev - term;
            }

            // q += c * x^k
            while q.len() < k + 1 {
                q.push(zero_c.clone());
            }
            let prev_q = std::mem::replace(&mut q[k], zero_c.clone());
            q[k] = prev_q + c;

            // The leading coefficient of r is now zero; trim it (and any others that
            // happened to cancel along the way).
            while r.last() == Some(&zero_c) {
                r.pop();
            }
        }

        (
            self.ring.element(q),
            self.ring.element(r),
        )
    }
}

/// `R[x]` is a Euclidean domain when `R` is a field. The unit part is the leading
/// coefficient (so canonical representatives are monic polynomials).
impl<R> EuclideanDomain for Rc<PolynomialRing<R>>
where
    R: Field,
    R::E: RingOps + Eq,
{
    fn unit_part(&self, x: &Self::E) -> Self::E {
        let lead = x
            .coefficients
            .last()
            .cloned()
            .unwrap_or_else(|| self.coeff_ring.identity());
        self.element(vec![lead])
    }

    fn unit_inverse(&self, u: &Self::E) -> Self::E {
        let c = u
            .coefficients
            .first()
            .expect("unit_inverse called on the zero polynomial");
        let c_inv = self
            .coeff_ring
            .invert(c)
            .expect("unit_inverse called on a non-unit polynomial");
        self.element(vec![c_inv])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::{Integer, Integers};

    fn int(n: i64) -> Integer {
        Integer::from(n)
    }

    fn zx() -> Rc<PolynomialRing<Integers>> {
        PolynomialRing::new(Integers::default())
    }

    fn poly(ring: &Rc<PolynomialRing<Integers>>, coeffs: Vec<i64>) -> Polynomial<Integers> {
        ring.element(coeffs.into_iter().map(int).collect::<Vec<_>>())
    }

    #[test]
    fn display_brackets_polynomial_coefficients() {
        let zx = zx();
        let zxy = PolynomialRing::new(zx.clone());
        let c = |coeffs: Vec<i64>| poly(&zx, coeffs);

        // (1 + 2x) + (3 + 4x)y — without brackets the two levels run together.
        let p = zxy.element(vec![c(vec![1, 2]), c(vec![3, 4])]);
        assert_eq!(format!("{}", p), "(1 + 2*x) + (3 + 4*x)*x");

        // Single-term coefficients are left alone.
        let q = zxy.element(vec![c(vec![7]), c(vec![0, 1])]);
        assert_eq!(format!("{}", q), "7 + (0 + 1*x)*x");
    }

    #[test]
    fn display_covers_zero_constant_linear_and_higher_terms() {
        let zx = zx();
        // Zero polynomial renders as "0", regardless of how it was constructed.
        assert_eq!(format!("{}", poly(&zx, vec![])), "0");
        assert_eq!(format!("{}", poly(&zx, vec![0, 0, 0])), "0");

        // Constant: no `x`.
        assert_eq!(format!("{}", poly(&zx, vec![7])), "7");

        // Linear: bare `x`, no exponent.
        assert_eq!(format!("{}", poly(&zx, vec![0, 1])), "0 + 1*x");
        assert_eq!(format!("{}", poly(&zx, vec![3, 2])), "3 + 2*x");

        // Degree ≥ 2 wraps the exponent in `{}` for LaTeX compatibility.
        assert_eq!(format!("{}", poly(&zx, vec![1, 2, 3])), "1 + 2*x + 3*x^{2}");
        assert_eq!(
            format!("{}", poly(&zx, vec![0, 0, 0, 4])),
            "0 + 0*x + 0*x^{2} + 4*x^{3}"
        );
    }

    #[test]
    fn new_strips_trailing_zero_coefficients() {
        let zx = zx();
        // Trailing zeros must be stripped so degree is well-defined and `==` works
        // structurally (both impls of Eq compare coefficient vectors directly).
        let a = poly(&zx, vec![1, 2, 0, 0]);
        let b = poly(&zx, vec![1, 2]);
        assert_eq!(a, b);

        // All-zero input collapses to the empty representation (the zero polynomial).
        let z = poly(&zx, vec![0, 0, 0]);
        assert_eq!(z, zx.zero());
    }
}
