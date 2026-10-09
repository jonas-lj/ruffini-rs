//! The field of fractions of a Euclidean domain: `Q` from `Z`, `F(x)` from `F[x]`.
//!
//! A fraction is kept in lowest terms, with the denominator's unit part divided out, so
//! equality is structural and each element has one representation: positive
//! denominators over `Z`, monic ones over `F[x]`.

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, DivRem, Division, Domain, EuclideanDomain, Field, Monoid,
    Ring, RingOps, SemiRing, Semigroup,
};
use num_bigint::BigInt;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use std::sync::Arc;

/// The field of fractions over `R`.
#[derive(Debug)]
pub struct FractionField<R: EuclideanDomain>
where
    R::E: RingOps + DivRem + Eq,
{
    domain: R,
}

/// An element of [`FractionField`], in lowest terms with a canonical denominator.
#[derive(Debug)]
pub struct Fraction<R: EuclideanDomain>
where
    R::E: RingOps + DivRem + Eq,
{
    numerator: R::E,
    denominator: R::E,
    ring: Arc<FractionField<R>>,
}

impl<R> FractionField<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    /// Construct the fractions over `domain`.
    pub fn new(domain: R) -> Arc<Self> {
        Arc::new(FractionField { domain })
    }

    /// The domain the numerators and denominators live in.
    pub fn base(&self) -> &R {
        &self.domain
    }

    /// `numerator / denominator`, reduced.
    ///
    /// # Panics
    /// If `denominator` is zero.
    pub fn fraction<N, M>(self: &Arc<Self>, numerator: N, denominator: M) -> Fraction<R>
    where
        N: Into<R::E>,
        M: Into<R::E>,
    {
        self.reduce(numerator.into(), denominator.into())
    }

    /// Divides out the gcd, then the denominator's unit part, which is what leaves one
    /// representation per element: the gcd alone would still allow `1/-2` beside `-1/2`.
    fn reduce(self: &Arc<Self>, numerator: R::E, denominator: R::E) -> Fraction<R> {
        let domain = &self.domain;
        assert!(
            denominator != domain.zero(),
            "zero denominator in a fraction"
        );

        // extended_gcd canonicalises the gcd itself, so this also fixes the sign of a
        // fraction whose numerator and denominator share no factor.
        let (gcd, _, _) = domain.extended_gcd(numerator.clone(), denominator.clone());
        let numerator = numerator.div_rem(&gcd).0;
        let denominator = denominator.div_rem(&gcd).0;

        let unit = domain.unit_inverse(&domain.unit_part(&denominator));
        Fraction {
            numerator: numerator * unit.clone(),
            denominator: denominator * unit,
            ring: Arc::clone(self),
        }
    }
}

impl<R> Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    /// The numerator, in lowest terms.
    pub fn numerator(&self) -> &R::E {
        &self.numerator
    }

    /// The denominator, in lowest terms and with its unit part divided out.
    pub fn denominator(&self) -> &R::E {
        &self.denominator
    }
}

impl<R> Clone for Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn clone(&self) -> Self {
        Fraction {
            numerator: self.numerator.clone(),
            denominator: self.denominator.clone(),
            ring: Arc::clone(&self.ring),
        }
    }
}

/// Structural, which the canonical form is what earns: `2/4` and `1/2` are the same
/// element and reduce to the same pair.
impl<R> PartialEq for Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        self.numerator == other.numerator && self.denominator == other.denominator
    }
}

impl<R> Eq for Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

/// `a/b + c/d = (ad + cb)/(bd)` and the same shape for subtraction, reduced afterwards.
macro_rules! fraction_additive_ops {
    ($($op:ident, $method:ident);* $(;)?) => {$(
        impl<R> $op<&Fraction<R>> for &Fraction<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
        {
            type Output = Fraction<R>;
            fn $method(self, rhs: &Fraction<R>) -> Fraction<R> {
                debug_assert!(
                    Arc::ptr_eq(&self.ring, &rhs.ring),
                    "Fraction operands belong to different fields"
                );
                let numerator = (self.numerator.clone() * rhs.denominator.clone())
                    .$method(rhs.numerator.clone() * self.denominator.clone());
                let denominator = self.denominator.clone() * rhs.denominator.clone();
                self.ring.reduce(numerator, denominator)
            }
        }

        impl<R> $op for Fraction<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
        {
            type Output = Fraction<R>;
            fn $method(self, rhs: Self) -> Fraction<R> {
                (&self).$method(&rhs)
            }
        }
    )*};
}
fraction_additive_ops!(Add, add; Sub, sub);

impl<R> Mul<&Fraction<R>> for &Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    type Output = Fraction<R>;
    fn mul(self, rhs: &Fraction<R>) -> Fraction<R> {
        debug_assert!(
            Arc::ptr_eq(&self.ring, &rhs.ring),
            "Fraction operands belong to different fields"
        );
        self.ring.reduce(
            self.numerator.clone() * rhs.numerator.clone(),
            self.denominator.clone() * rhs.denominator.clone(),
        )
    }
}

impl<R> Mul for Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    type Output = Fraction<R>;
    fn mul(self, rhs: Self) -> Fraction<R> {
        (&self).mul(&rhs)
    }
}

forward_ref_binops!(
    Fraction<R>,
    { R: EuclideanDomain, R::E: RingOps + DivRem + Eq, },
    Add, add; Sub, sub; Mul, mul
);

int_operand_ops!(
    Fraction<R>,
    { R: EuclideanDomain, R::E: RingOps + DivRem + Eq, },
    i64,
    BigInt
);

scalar_operand_ops!(
    Fraction<R>,
    { R: EuclideanDomain, R::E: RingOps + DivRem + Eq, },
    i64,
    BigInt
);

element_neg!(
    Fraction<R>,
    { R: EuclideanDomain, R::E: RingOps + DivRem + Eq, }
);

element_pow!(
    Fraction<R>,
    { R: EuclideanDomain, R::E: RingOps + DivRem + Eq, }
);

/// Compound assignment, forwarding to the borrowed operators.
macro_rules! fraction_assign_ops {
    ($($op:ident, $method:ident, $base_method:ident);* $(;)?) => {$(
        impl<R> $op<&Fraction<R>> for Fraction<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
        {
            fn $method(&mut self, rhs: &Fraction<R>) {
                *self = (&*self).$base_method(rhs);
            }
        }

        impl<R> $op for Fraction<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
        {
            fn $method(&mut self, rhs: Self) {
                *self = (&*self).$base_method(&rhs);
            }
        }
    )*};
}
fraction_assign_ops!(AddAssign, add_assign, add; MulAssign, mul_assign, mul);

/// Exact division, which is total away from zero. The remainder is always zero, which
/// is what makes a field a Euclidean domain.
impl<R> DivRem for Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    /// # Panics
    /// If `divisor` is zero.
    fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let quotient = self.ring.reduce(
            self.numerator.clone() * divisor.denominator.clone(),
            self.denominator.clone() * divisor.numerator.clone(),
        );
        let zero = self.ring.zero();
        (quotient, zero)
    }
}

impl<R> fmt::Display for Fraction<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.denominator == self.ring.domain.identity() {
            write!(f, "{}", self.numerator)
        } else {
            write!(f, "{}/{}", self.numerator, self.denominator)
        }
    }
}

impl<R> Domain for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    type E = Fraction<R>;

    /// An element of the base domain, as [`QuotientRing`](crate::structures::QuotientRing)
    /// also takes. Not the base's own `Repr`: over `F[x]` that would be a coefficient
    /// vector, where what one has in hand is a polynomial.
    type Repr = R::E;

    /// `value / 1`, so an integer literal denotes the fraction it names.
    fn element<T: Into<R::E>>(&self, value: T) -> Self::E {
        self.reduce(value.into(), self.domain.identity())
    }
}

impl<R> Semigroup for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

impl<R> Monoid for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn identity(&self) -> Self::E {
        Fraction {
            numerator: self.domain.identity(),
            denominator: self.domain.identity(),
            ring: Arc::clone(self),
        }
    }
}

impl<R> CommutativeMonoid for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn zero(&self) -> Self::E {
        Fraction {
            numerator: self.domain.zero(),
            denominator: self.domain.identity(),
            ring: Arc::clone(self),
        }
    }
}

impl<R> AdditiveGroup for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

impl<R> SemiRing for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

impl<R> Ring for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

/// Every nonzero element is a unit, so a gcd is only ever zero or one and the canonical
/// associate of `x` is one.
impl<R> EuclideanDomain for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    type Reducer = Division<Self::E>;

    fn unit_part(&self, x: &Self::E) -> Self::E {
        if self.is_zero(x) {
            self.identity()
        } else {
            x.clone()
        }
    }

    fn unit_inverse(&self, u: &Self::E) -> Self::E {
        self.invert(u).expect("unit_inverse of zero")
    }
}

impl<R> Field for Arc<FractionField<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn invert(&self, x: &Self::E) -> Option<Self::E> {
        if self.is_zero(x) {
            return None;
        }
        Some(self.reduce(x.denominator.clone(), x.numerator.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::{Integer, Integers};
    use crate::polynomials::{Polynomial, PolynomialRing, RingExt};

    fn q() -> Arc<FractionField<Integers>> {
        Integers::rationals()
    }

    #[test]
    fn fractions_are_kept_in_lowest_terms_with_a_canonical_sign() {
        let q = q();

        // The same element, however it is written down.
        assert_eq!(q.fraction(1, 2), q.fraction(2, 4));
        assert_eq!(q.fraction(1, 2), q.fraction(-3, -6));
        assert_eq!(q.fraction(-1, 2), q.fraction(1, -2));
        assert_ne!(q.fraction(1, 2), q.fraction(1, 3));

        // The denominator is the positive one of the pair, so display is unambiguous.
        let half = q.fraction(1, -2);
        assert_eq!(half.numerator(), &Integer::from(-1));
        assert_eq!(half.denominator(), &Integer::from(2));
        assert_eq!(format!("{half}"), "-1/2");

        // Zero and integers collapse to a unit denominator.
        assert_eq!(q.fraction(0, 5), q.zero());
        assert_eq!(q.fraction(6, 3), q.element(2));
        assert_eq!(format!("{}", q.element(2)), "2");
        assert_eq!(q.fraction(5, 5), q.identity());
    }

    #[test]
    #[should_panic(expected = "zero denominator")]
    fn rejects_a_zero_denominator() {
        q().fraction(1, 0);
    }

    #[test]
    fn arithmetic_is_the_usual_rational_arithmetic() {
        let q = q();

        assert_eq!(q.fraction(1, 2) + q.fraction(1, 3), q.fraction(5, 6));
        assert_eq!(q.fraction(1, 2) - q.fraction(1, 3), q.fraction(1, 6));
        assert_eq!(q.fraction(2, 3) * q.fraction(3, 4), q.fraction(1, 2));
        assert_eq!(q.fraction(1, 2) + q.fraction(-1, 2), q.zero());

        // Borrowed and mixed operands agree with the owned ones.
        let (a, b) = (q.fraction(3, 4), q.fraction(5, 6));
        assert_eq!(&a + &b, a.clone() + b.clone());
        assert_eq!(a.clone() + &b, a.clone() + b.clone());
        assert_eq!(&a * b.clone(), a.clone() * b.clone());

        // Integers on either side, and powers.
        assert_eq!(q.fraction(1, 2) * 4, q.element(2));
        assert_eq!(2 * q.fraction(1, 4), q.fraction(1, 2));
        assert_eq!(1 - q.fraction(1, 4), q.fraction(3, 4));
        assert_eq!(q.fraction(2, 3).pow(3), q.fraction(8, 27));
    }

    #[test]
    fn every_nonzero_fraction_is_invertible() {
        let q = q();

        assert_eq!(q.invert(&q.fraction(3, 4)), Some(q.fraction(4, 3)));
        assert_eq!(q.invert(&q.fraction(-3, 4)), Some(q.fraction(-4, 3)));
        assert_eq!(q.invert(&q.zero()), None);
        assert_eq!(q.inverse(5), Some(q.fraction(1, 5)));

        for (n, d) in [(1, 2), (-3, 7), (22, 7), (5, 1)] {
            let x = q.fraction(n, d);
            assert_eq!(&x * &q.invert(&x).unwrap(), q.identity());
        }

        // Division leaves no remainder, which is what makes a field Euclidean.
        let (quotient, remainder) = q.fraction(1, 2).div_rem(&q.fraction(3, 4));
        assert_eq!(quotient, q.fraction(2, 3));
        assert!(q.is_zero(&remainder));
    }

    #[test]
    fn rational_functions_come_out_with_monic_denominators() {
        // F_7(x): the unit part of a polynomial is its leading coefficient, so
        // canonicalising the denominator makes it monic rather than positive.
        let f7 = Integers::modulo(7);
        let f7x = f7.polynomials();
        let field = f7x.fractions();
        let x = f7x.indeterminate();

        // 2x / 4 is x / 2 in lowest terms, and then x * 4 over 1 once 2 is inverted.
        let f = field.fraction(x.clone() * 2, f7x.from_integer(4));
        assert_eq!(f.denominator(), &f7x.identity());
        assert_eq!(f.numerator(), &(x.clone() * 4));

        // 1/x is a genuine fraction, and x * (1/x) = 1.
        let inverse_x = field.invert(&field.element(x.clone())).unwrap();
        assert_eq!(inverse_x.numerator(), &f7x.identity());
        assert_eq!(inverse_x.denominator(), &x);
        assert_eq!(field.element(x.clone()) * inverse_x, field.identity());

        // (x^2 - 1)/(x - 1) reduces to x + 1.
        let numerator = x.pow(2) - 1;
        let denominator = x.clone() - 1;
        let reduced = field.fraction(numerator, denominator);
        assert_eq!(reduced, field.element(x.clone() + 1));
    }

    #[test]
    fn interpolation_over_q_gives_exact_rational_coefficients() {
        // The parabola through (0,0), (1,1), (2,4) is x^2, and through (0,0), (1,1),
        // (2,3) it is x^2/2 + x/2 - which needs a halving that Z cannot do.
        let q = q();
        let qx = q.polynomials();
        let xs: Vec<_> = (0..3).map(|n| q.element(n)).collect();

        let squares: Vec<_> = [0, 1, 4].iter().map(|n| q.element(*n)).collect();
        let p = crate::interpolation::interpolate(&qx, &xs, &squares).unwrap();
        assert_eq!(p, qx.indeterminate().pow(2));

        let triangular: Vec<_> = [0, 1, 3].iter().map(|n| q.element(*n)).collect();
        let p: Polynomial<Arc<FractionField<Integers>>> =
            crate::interpolation::interpolate(&qx, &xs, &triangular).unwrap();

        // x^2/2 + x/2, which has no counterpart over Z.
        let half = q.fraction(1, 2);
        assert_eq!(
            p,
            qx.element(vec![q.zero(), half.clone(), half])
        );
        // n(n+1)/2 at n = 4 is 10.
        assert_eq!(p.evaluate(&q.element(4)), q.element(10));
    }

    #[test]
    fn matrices_over_q_invert_exactly() {
        use crate::matrices::Matrix;
        let q = q();
        let m = Matrix::new(
            q.clone(),
            2,
            2,
            vec![q.element(1), q.element(2), q.element(3), q.element(4)],
        );

        // det = -2, so the inverse is [[-2, 1], [3/2, -1/2]].
        assert_eq!(m.determinant(), q.element(-2));
        let inverse = m.inverse().expect("determinant is nonzero");
        assert_eq!(inverse.get(0, 0), &q.element(-2));
        assert_eq!(inverse.get(1, 0), &q.fraction(3, 2));
        assert_eq!(inverse.get(1, 1), &q.fraction(-1, 2));

        let identity = q.matrices(2).identity();
        assert_eq!(&m * &inverse, identity);
    }

    #[test]
    fn a_fraction_field_can_be_stacked_on_a_polynomial_ring_over_fractions() {
        // Q(x), reached as the fractions of Q[x] - two levels of this construction.
        let q = q();
        let qx: Arc<PolynomialRing<Arc<FractionField<Integers>>>> = q.polynomials();
        let field = qx.fractions();
        let x = qx.indeterminate();

        let half = field.fraction(qx.element(vec![q.fraction(1, 2)]), qx.identity());
        assert_eq!(half.clone() + half, field.element(qx.identity()));
        assert_eq!(
            field.invert(&field.element(x.clone() * 2)).unwrap(),
            field.fraction(qx.identity(), x.clone() * 2)
        );
    }
}
