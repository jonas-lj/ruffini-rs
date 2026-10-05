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
use std::sync::Arc;

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
    ring: Arc<PolynomialRing<R>>,
}

/// Puts [`polynomials`](RingExt::polynomials) on every ring, so `structures` does not
/// have to name this module.
pub trait RingExt: Ring + Clone + Sized
where
    Self::E: RingOps + Eq,
{
    /// The polynomial ring `Self[x]`.
    fn polynomials(&self) -> Arc<PolynomialRing<Self>> {
        PolynomialRing::new(self.clone())
    }

    /// The ring of `n x n` matrices over `Self`.
    fn matrices(&self, n: usize) -> Arc<crate::matrices::MatrixRing<Self>> {
        crate::matrices::MatrixRing::new(self.clone(), n)
    }

    /// The polynomial ring `Self[x_0, .., x_{n-1}]`.
    ///
    /// Unlike repeated [`polynomials`](RingExt::polynomials), the variable count is a
    /// run-time value: nesting puts the arity in the type, this does not.
    fn multi_polynomials(
        &self,
        variables: usize,
    ) -> Arc<crate::multivariate::MultivariatePolynomialRing<Self>> {
        crate::multivariate::MultivariatePolynomialRing::new(self.clone(), variables)
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
    /// Construct `R[x]`, wrapped in an [`Arc`] so polynomials can share the ring handle.
    pub fn new(coeff_ring: R) -> Arc<Self> {
        Arc::new(PolynomialRing { coeff_ring })
    }

    /// The underlying coefficient ring.
    pub fn coefficients(&self) -> &R {
        &self.coeff_ring
    }

    /// The constant polynomial `c`, for lifting a coefficient into the ring. An
    /// operator cannot do this - nothing rules out `R::E` being `Polynomial<R>`, so
    /// `Polynomial<R>: Sub<R::E>` is read as overlapping with polynomial subtraction.
    pub fn constant(self: &Arc<Self>, c: R::E) -> Polynomial<R> {
        self.element(vec![c])
    }

    /// The indeterminate `x`, so polynomials can be written the way they are read:
    /// `x.pow(n) - 1` rather than a padded coefficient vector.
    pub fn indeterminate(self: &Arc<Self>) -> Polynomial<R> {
        self.element(vec![self.coeff_ring.zero(), self.coeff_ring.identity()])
    }
}

impl<R> Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    /// The coefficient of the highest-degree term, or [`None`] for the zero polynomial.
    pub fn lead(&self) -> Option<&R::E> {
        self.coefficients.last()
    }

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
            ring: Arc::clone(&self.ring),
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
            Arc::ptr_eq(&self.ring, &other.ring),
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
            Arc::ptr_eq(&self.ring, &rhs.ring),
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
            Arc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        let result: Vec<R::E> = self
            .coefficients
            .into_iter()
            .zip_longest(rhs.coefficients)
            .map(|pair| match pair {
                EitherOrBoth::Both(x, y) => x - y,
                EitherOrBoth::Left(x) => x,
                // Owned Sub cannot take a borrowed-operand bound without RingOps
                // dragging it through nested polynomials, so zero is replaced per term.
                EitherOrBoth::Right(y) => -y,
            })
            .collect();
        self.ring.element(result)
    }
}

/// Shortest operand at which the `Mul` impls switch from the schoolbook product to
/// [`karatsuba`], and equally the length at which the recursion stops again.
///
/// Measured over `Z[x]` and `F_40961[x]` by sweeping the cutoff against lengths from 12
/// to 1024. Thirty-two was the fastest cutoff at every length tried - 1.4x the
/// schoolbook product at length 32, rising to 3.9x at 1024 - and the shortest length at
/// which recursing wins over both rings rather than only over the one whose
/// coefficients are dearer. Below it the recursion's own allocations outweigh the
/// multiplication it saves, by as much as 30% at length 16.
const KARATSUBA_THRESHOLD: usize = 32;

/// `a * b` in `O(n^1.585)`, by Karatsuba's three half-length products in place of four.
///
/// Karatsuba and Ofman, "Multiplication of Many-Digital Numbers by Automatic Computers",
/// Proceedings of the USSR Academy of Sciences 145 (1962).
///
/// Asks nothing of the ring, unlike a transform-based product, which needs a root of
/// unity the ring may not have.
///
/// The operands need not be the same length. A lopsided pair splits off an empty half,
/// which costs one product more than the schoolbook form rather than giving a wrong
/// answer; the threshold check is what keeps that from mattering.
fn karatsuba<R>(ring: &R, a: &[R::E], b: &[R::E]) -> Vec<R::E>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    let (n, m) = (a.len(), b.len());
    if n.min(m) < KARATSUBA_THRESHOLD {
        return schoolbook(ring, a, b);
    }

    let half = n.max(m).div_ceil(2);
    let (a_low, a_high) = a.split_at(half.min(n));
    let (b_low, b_high) = b.split_at(half.min(m));

    let low = karatsuba(ring, a_low, b_low);
    let high = karatsuba(ring, a_high, b_high);
    let mixed = karatsuba(ring, &sum(ring, a_low, a_high), &sum(ring, b_low, b_high));

    // The middle coefficient is `mixed - low - high`, accumulated in place rather than
    // built as its own polynomial first.
    let mut result = Vec::new();
    add_at(ring, &mut result, &low, 0);
    sub_at(ring, &mut result, &low, half);
    add_at(ring, &mut result, &high, 2 * half);
    sub_at(ring, &mut result, &high, half);
    add_at(ring, &mut result, &mixed, half);
    result
}

/// The schoolbook product, gathering each output coefficient independently of the rest.
fn schoolbook<R>(ring: &R, a: &[R::E], b: &[R::E]) -> Vec<R::E>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    if a.is_empty() || b.is_empty() {
        return Vec::new();
    }
    let (n, m) = (a.len(), b.len());
    let zero = ring.zero();
    (0..n + m - 1)
        .map(|k| {
            (k.saturating_sub(m - 1)..=k.min(n - 1))
                .map(|i| a[i].clone() * b[k - i].clone())
                .fold(zero.clone(), |sum, term| sum + term)
        })
        .collect()
}

fn sum<R>(ring: &R, a: &[R::E], b: &[R::E]) -> Vec<R::E>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    let mut out = Vec::new();
    add_at(ring, &mut out, a, 0);
    add_at(ring, &mut out, b, 0);
    out
}

/// `target[shift..] += addend`, lengthening `target` if the addend reaches past its end.
/// Growing rather than asserting, so a miscounted bound cannot silently truncate a term.
fn add_at<R>(ring: &R, target: &mut Vec<R::E>, addend: &[R::E], shift: usize)
where
    R: Ring,
    R::E: RingOps + Eq,
{
    grow(ring, target, shift + addend.len());
    for (slot, term) in target[shift..].iter_mut().zip(addend) {
        *slot += term.clone();
    }
}

/// `target[shift..] -= subtrahend`. There is no `SubAssign` in [`RingOps`], so each slot
/// is taken out and put back.
fn sub_at<R>(ring: &R, target: &mut Vec<R::E>, subtrahend: &[R::E], shift: usize)
where
    R: Ring,
    R::E: RingOps + Eq,
{
    grow(ring, target, shift + subtrahend.len());
    for (slot, term) in target[shift..].iter_mut().zip(subtrahend) {
        let left = std::mem::replace(slot, ring.zero());
        *slot = left - term.clone();
    }
}

fn grow<R>(ring: &R, target: &mut Vec<R::E>, len: usize)
where
    R: Ring,
    R::E: RingOps + Eq,
{
    if target.len() < len {
        target.resize(len, ring.zero());
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
            Arc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        if self.coefficients.is_empty() || rhs.coefficients.is_empty() {
            return Polynomial {
                coefficients: Vec::new(),
                ring: self.ring,
            };
        }
        let (a, b) = (&self.coefficients, &rhs.coefficients);
        if a.len().min(b.len()) >= KARATSUBA_THRESHOLD {
            return self.ring.element(karatsuba(&self.ring.coeff_ring, a, b));
        }
        // Each output coefficient is the sum over i + j == k, gathered independently of
        // the others, so the outer map has no shared state to contend over.
        let result = schoolbook(&self.ring.coeff_ring, a, b);
        self.ring.element(result)
    }
}

/// Coefficient-wise addition of two borrowed polynomials. The shorter one is padded
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
            Arc::ptr_eq(&self.ring, &rhs.ring),
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
            Arc::ptr_eq(&self.ring, &rhs.ring),
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
            Arc::ptr_eq(&self.ring, &rhs.ring),
            "Polynomial operands belong to different rings"
        );
        if self.coefficients.is_empty() || rhs.coefficients.is_empty() {
            return Polynomial {
                coefficients: Vec::new(),
                ring: Arc::clone(&self.ring),
            };
        }
        let (a, b) = (&self.coefficients, &rhs.coefficients);
        if a.len().min(b.len()) >= KARATSUBA_THRESHOLD {
            return self.ring.element(karatsuba(&self.ring.coeff_ring, a, b));
        }
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

scalar_operand_ops!(
    Polynomial<R>,
    { R: Ring, R::E: RingOps + Eq, },
    i64,
    BigInt
);

element_pow!(Polynomial<R>, { R: Ring, R::E: RingOps + Eq, });
element_neg!(Polynomial<R>, { R: Ring, R::E: RingOps + Eq, });

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
/// only an `Arc` bump, so the left operand is still not cloned.
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
                    ring: Arc::clone(&self.ring),
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
                    ring: Arc::clone(&self.ring),
                };
                let lhs = std::mem::replace(self, placeholder);
                *self = lhs.$base_method(rhs);
            }
        }
    )*};
}
polynomial_assign_ops!(AddAssign, add_assign, add; MulAssign, mul_assign, mul);


impl<R> Domain for Arc<PolynomialRing<R>>
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
            ring: Arc::clone(self),
        }
    }
}

impl<R> Semigroup for Arc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Monoid for Arc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn identity(&self) -> Self::E {
        self.element(vec![self.coeff_ring.identity()])
    }
}

impl<R> CommutativeMonoid for Arc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn zero(&self) -> Self::E {
        self.element(Vec::new())
    }
}

impl<R> AdditiveGroup for Arc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> SemiRing for Arc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Ring for Arc<PolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Polynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    /// Long division, given the inverse of the divisor's leading coefficient.
    ///
    /// Nothing else needs inverting, which is what lets division work over a ring
    /// rather than only a field.
    fn divide_by(&self, divisor: &Self, lead_inverse: &R::E) -> (Self, Self) {
        debug_assert!(
            Arc::ptr_eq(&self.ring, &divisor.ring),
            "Polynomial operands belong to different rings"
        );
        let zero_c = self.ring.coeff_ring.zero();
        let degree = divisor.coefficients.len() - 1;

        let mut r: Vec<R::E> = self.coefficients.clone();
        let mut q: Vec<R::E> = Vec::new();

        while r.len() > degree {
            // c = leading(r) / leading(divisor)
            let c = r.last().unwrap().clone() * lead_inverse.clone();
            let k = r.len() - 1 - degree;

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

        (self.ring.element(q), self.ring.element(r))
    }

    /// Division by a monic divisor, which works over any ring.
    ///
    /// A monic leading coefficient is its own inverse, so unlike [`DivRem::div_rem`]
    /// this needs no field. Reduction modulo `x^n - 1`, say, is fine over `Z`.
    ///
    /// # Panics
    /// If the divisor is zero or is not monic.
    pub fn div_rem_monic(&self, divisor: &Self) -> (Self, Self) {
        let one = self.ring.coeff_ring.identity();
        assert!(
            divisor.lead().expect("division by zero polynomial") == &one,
            "divisor is not monic"
        );
        self.divide_by(divisor, &one)
    }
}

/// Polynomial long division. Requires `R: Field` so leading coefficients can be
/// inverted; see [`Polynomial::div_rem_monic`] for the case that does not.
/// Panics if the divisor is the zero polynomial.
impl<R> DivRem for Polynomial<R>
where
    R: Field,
    R::E: RingOps + Eq,
{
    fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let lead = divisor.lead().expect("division by zero polynomial");
        let lead_inverse = self
            .ring
            .coeff_ring
            .invert(lead)
            .expect("leading coefficient must be invertible in a field");
        self.divide_by(divisor, &lead_inverse)
    }
}

/// `R[x]` is a Euclidean domain when `R` is a field. The unit part is the leading
/// coefficient (so canonical representatives are monic polynomials).
impl<R> EuclideanDomain for Arc<PolynomialRing<R>>
where
    R: Field,
    R::E: RingOps + Eq,
{
    fn unit_part(&self, x: &Self::E) -> Self::E {
        let lead = x
            .lead()
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

    fn zx() -> Arc<PolynomialRing<Integers>> {
        PolynomialRing::new(Integers::default())
    }

    fn poly(ring: &Arc<PolynomialRing<Integers>>, coeffs: Vec<i64>) -> Polynomial<Integers> {
        ring.element(coeffs.into_iter().map(int).collect::<Vec<_>>())
    }

    /// The convolution straight from its definition, to check the product against
    /// without going through the library's own schoolbook path.
    fn convolve(a: &[i64], b: &[i64]) -> Vec<i64> {
        if a.is_empty() || b.is_empty() {
            return Vec::new();
        }
        let mut out = vec![0i64; a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            for (j, y) in b.iter().enumerate() {
                out[i + j] += x * y;
            }
        }
        out
    }

    /// Deterministic pseudo-random coefficients in `-4..=4`, so zeros occur and the
    /// trimming of leading zeros gets exercised.
    fn coefficients(len: usize, seed: u64) -> Vec<i64> {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) % 9) as i64 - 4
            })
            .collect()
    }

    #[test]
    fn karatsuba_agrees_with_the_convolution_either_side_of_the_threshold() {
        let zx = zx();
        let t = KARATSUBA_THRESHOLD;

        // Shapes straddling the threshold, including lopsided and equal-length pairs,
        // and sizes that force an odd split.
        let shapes = [
            (1, 1),
            (t - 1, t - 1),
            (t - 1, t),
            (t, t - 1),
            (t, t),
            (t + 1, t),
            (t + 1, t + 1),
            (2 * t, t),
            (2 * t + 1, t + 3),
            (4 * t + 5, 4 * t + 5),
            (6 * t, t + 1),
        ];

        for (i, (n, m)) in shapes.into_iter().enumerate() {
            let (left, right) = (
                coefficients(n, i as u64 + 1),
                coefficients(m, i as u64 + 99),
            );
            let expected = {
                let mut c = convolve(&left, &right);
                while c.last() == Some(&0) {
                    c.pop();
                }
                poly(&zx, c)
            };
            let (a, b) = (poly(&zx, left), poly(&zx, right));

            // Both operand forms, and multiplication is commutative here.
            assert_eq!(&a * &b, expected, "&a * &b at {n} x {m}");
            assert_eq!(a.clone() * b.clone(), expected, "a * b at {n} x {m}");
            assert_eq!(&b * &a, expected, "&b * &a at {n} x {m}");
            assert_eq!(a.clone() * &b, expected, "a * &b at {n} x {m}");
        }
    }

    #[test]
    fn karatsuba_handles_zero_and_sparse_operands() {
        let zx = zx();
        let t = KARATSUBA_THRESHOLD;

        // A single high term times a dense operand: every intermediate half is empty on
        // one side, which is the lopsided split.
        let mut sparse = vec![0i64; 4 * t];
        sparse[4 * t - 1] = 3;
        let dense = coefficients(4 * t, 7);

        let expected = {
            let mut c = convolve(&sparse, &dense);
            while c.last() == Some(&0) {
                c.pop();
            }
            poly(&zx, c)
        };
        assert_eq!(
            &poly(&zx, sparse.clone()) * &poly(&zx, dense.clone()),
            expected
        );

        // Zero annihilates whatever its length suggests.
        let zeros = poly(&zx, vec![0i64; 4 * t]);
        assert_eq!(zeros, zx.zero());
        assert_eq!(&zeros * &poly(&zx, dense), zx.zero());

        // And the identity is still the identity above the threshold.
        let one = zx.identity();
        let big = poly(&zx, coefficients(4 * t, 11));
        assert_eq!(&big * &one, big);
    }

    #[test]
    fn karatsuba_works_over_a_quotient_ring_too() {
        // Coefficients that reduce, so the recursion's additions and subtractions have
        // to stay inside the ring rather than in Z.
        let f7 = Integers::modulo(7);
        let f7x = f7.polynomials();
        let t = KARATSUBA_THRESHOLD;

        let (left, right) = (coefficients(3 * t, 5), coefficients(3 * t, 13));
        let lift = |c: &[i64]| f7x.element(c.iter().map(|n| f7.element(*n)).collect::<Vec<_>>());

        let expected = lift(&convolve(&left, &right));
        assert_eq!(&lift(&left) * &lift(&right), expected);

        // x^n * x^m = x^(n+m), which pins the shifts down independently of the sums.
        let x = f7x.indeterminate();
        assert_eq!(&x.pow(3 * t) * &x.pow(2 * t), x.pow(5 * t));
    }

    #[test]
    fn an_indeterminate_and_pow_replace_the_coefficient_vector() {
        let zx = zx();
        let x = zx.indeterminate();

        assert_eq!(x, poly(&zx, vec![0, 1]));
        assert_eq!(x.pow(0), zx.identity());
        assert_eq!(x.pow(1), x);
        assert_eq!(x.pow(3), poly(&zx, vec![0, 0, 0, 1]));

        // x^7 - 1, against the padded vector it used to take to write.
        let mut coefficients = vec![-1i64];
        coefficients.resize(7, 0);
        coefficients.push(1);
        assert_eq!(x.pow(7) - 1, poly(&zx, coefficients));

        // pow is not just shifting: (1 + x)^3 = 1 + 3x + 3x^2 + x^3.
        assert_eq!((1 + x.clone()).pow(3), poly(&zx, vec![1, 3, 3, 1]));
    }

    #[test]
    fn a_scalar_on_the_left_matches_the_same_scalar_embedded() {
        let zx = zx();
        let x = zx.indeterminate();
        let one = zx.identity();

        assert_eq!(2 * x.pow(3), x.pow(3) * 2);
        assert_eq!(3 + x.clone(), poly(&zx, vec![3, 1]));

        // Subtraction is the one that does not commute, so it pins the operand order.
        assert_eq!(1 - x.clone(), poly(&zx, vec![1, -1]));
        assert_eq!(x.clone() - 1, poly(&zx, vec![-1, 1]));
        assert_eq!(1 - x.clone(), one - x.clone());

        // BigInt reaches the same impls as i64.
        assert_eq!(BigInt::from(2) * x.clone(), 2 * x.clone());
        assert_eq!(BigInt::from(1) - x.clone(), 1 - x.clone());
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
