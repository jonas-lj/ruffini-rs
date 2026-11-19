//! Montgomery curves `y^2 = x^3 + Ax^2 + x`, with arithmetic on `x` alone.
//!
//! The `B` of the general form is taken to be one. What the shape buys is cheap
//! arithmetic: a point is kept as `(X : Z)` standing for `x = X/Z`, both of `P` and
//! `-P` at once, and doubling and differential addition never mention `y`.
//!
//! That is also why `x` needs no checking. Every `x` in the field is the coordinate of
//! a point of `E_A` or of its quadratic twist, as `x^3 + Ax^2 + x` is a square or not,
//! and the same formulas serve both, so a point of given order is found by picking an
//! `x` rather than solving for one.

use crate::elliptic::Curve;
use crate::structures::{Field, RingOps};
use num_bigint::BigInt;
use std::fmt;
use std::ops::{Div, Mul, MulAssign};
use std::sync::Arc;

/// The curve `y^2 = x^3 + Ax^2 + x` over a field of characteristic above three.
#[derive(Debug)]
pub struct MontgomeryCurve<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    field: F,
    a: F::E,
    /// `(A + 2)/4`, which every doubling asks for.
    a24: F::E,
}

/// A point by its `x` coordinate alone, as `(X : Z)` with `x = X/Z` and `Z = 0` the
/// point at infinity. It stands for `P` and `-P` together, which share an `x`.
#[derive(Debug)]
pub struct XPoint<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    x: F::E,
    z: F::E,
}

impl<F> MontgomeryCurve<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    /// The curve with coefficient `a`, or [`None`] if it is singular, which is when
    /// `A^2 = 4`.
    ///
    /// # Panics
    /// If the characteristic is 2 or 3, which this form rules out.
    pub fn new(field: F, a: F::E) -> Option<Arc<Self>> {
        assert!(
            !field.is_zero(&field.from_integer(6)),
            "a Montgomery curve needs a characteristic above three"
        );
        if a.clone() * a.clone() == field.from_integer(4) {
            return None;
        }
        let quarter = field
            .invert(&field.from_integer(4))
            .expect("four is a unit away from characteristic two");
        let a24 = (a.clone() + field.from_integer(2)) * quarter;
        Some(Arc::new(MontgomeryCurve { field, a, a24 }))
    }

    pub fn field(&self) -> &F {
        &self.field
    }

    /// The `A` of `y^2 = x^3 + Ax^2 + x`.
    pub fn coefficient(&self) -> &F::E {
        &self.a
    }

    /// `x^3 + Ax^2 + x`, the right-hand side. A square means `x` is on the curve, and
    /// anything else means it is on the quadratic twist.
    pub fn evaluate(&self, x: &F::E) -> F::E {
        let square = x.clone() * x.clone();
        square.clone() * x.clone() + self.a.clone() * square + x.clone()
    }

    /// The point `(x : 1)`.
    pub fn point(&self, x: F::E) -> XPoint<F> {
        XPoint {
            x,
            z: self.field.identity(),
        }
    }

    /// The point at infinity, `(1 : 0)`.
    pub fn infinity(&self) -> XPoint<F> {
        XPoint {
            x: self.field.identity(),
            z: self.field.zero(),
        }
    }

    /// Whether `p` is the point at infinity.
    pub fn is_infinity(&self, p: &XPoint<F>) -> bool {
        self.field.is_zero(&p.z)
    }

    /// `X/Z`, or [`None`] at infinity.
    pub fn affine_x(&self, p: &XPoint<F>) -> Option<F::E> {
        self.field.invert(&p.z).map(|inverse| p.x.clone() * inverse)
    }

    /// `[2]P`, by the doubling of Montgomery's ladder.
    ///
    /// With `s = (X + Z)^2` and `d = (X - Z)^2`, their difference is `4XZ`, and
    /// `[2]P = (s*d : 4XZ*(d + a24*4XZ))`.
    pub fn double(&self, p: &XPoint<F>) -> XPoint<F> {
        let sum = p.x.clone() + p.z.clone();
        let difference = p.x.clone() - p.z.clone();
        let (s, d) = (sum.clone() * sum, difference.clone() * difference);
        let t = s.clone() - d.clone();
        XPoint {
            x: s * d.clone(),
            z: t.clone() * (d + self.a24.clone() * t),
        }
    }

    /// `P + Q`, which on `x` alone needs `P - Q` as well.
    ///
    /// Nothing checks that `difference` is `P - Q`. Given anything else this returns
    /// some other point of the curve rather than failing.
    pub fn differential_add(
        &self,
        p: &XPoint<F>,
        q: &XPoint<F>,
        difference: &XPoint<F>,
    ) -> XPoint<F> {
        let u = (p.x.clone() - p.z.clone()) * (q.x.clone() + q.z.clone());
        let v = (p.x.clone() + p.z.clone()) * (q.x.clone() - q.z.clone());
        let (sum, gap) = (u.clone() + v.clone(), u - v);
        XPoint {
            x: difference.z.clone() * (sum.clone() * sum),
            z: difference.x.clone() * (gap.clone() * gap),
        }
    }

    /// `[k]P`, by Montgomery's ladder, which carries `([m]P, [m+1]P)` and so always
    /// knows the difference `P` that the addition needs. A negative `k` gives the same
    /// point as `|k|`, since `x` cannot tell them apart.
    pub fn ladder<T: Into<BigInt>>(&self, k: T, p: &XPoint<F>) -> XPoint<F> {
        let (_, magnitude) = k.into().into_parts();
        let bits = magnitude.bits();
        if bits == 0 || self.is_infinity(p) {
            return self.infinity();
        }
        // The differential addition divides the plane by `X` of the difference, so at
        // `(0 : 1)` its `Z` vanishes whatever the input. That point has order two,
        // which settles `[k]P` without the ladder.
        if self.field.is_zero(&p.x) {
            return if magnitude.bit(0) {
                p.clone()
            } else {
                self.infinity()
            };
        }

        let (mut lower, mut upper) = (p.clone(), self.double(p));
        for i in (0..bits - 1).rev() {
            if magnitude.bit(i) {
                lower = self.differential_add(&lower, &upper, p);
                upper = self.double(&upper);
            } else {
                upper = self.differential_add(&lower, &upper, p);
                lower = self.double(&lower);
            }
        }
        lower
    }
}

impl<F> MontgomeryCurve<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq + Div<Output = F::E> + for<'x> MulAssign<&'x F::E>,
    i64: Mul<F::E, Output = F::E>,
{
    /// The j-invariant, `256 (A^2 - 3)^3 / (A^2 - 4)`.
    pub fn j_invariant(&self) -> F::E {
        let field = &self.field;
        let square = self.a.clone() * self.a.clone();
        let numerator =
            field.from_integer(256) * field.pow(&(square.clone() - field.from_integer(3)), 3);
        // Nonzero: `new` rejects the curves where it would vanish.
        numerator / (square - field.from_integer(4))
    }

    /// The same curve in short Weierstrass form, which `x = X - A/3` takes it to:
    /// `Y^2 = X^3 + (1 - A^2/3) X + (2A^3/27 - A/3)`.
    pub fn to_weierstrass(&self) -> Arc<Curve<F>> {
        let field = &self.field;
        let third = self.third();
        let square = self.a.clone() * self.a.clone();
        let cube = square.clone() * self.a.clone();

        let a = field.identity() - square * third.clone();
        let b = field.from_integer(2) * cube * field.pow(&third, 3) - self.a.clone() * third;
        Curve::new(field.clone(), a, b).expect("the substitution keeps the curve nonsingular")
    }

    /// Where `x` goes in the Weierstrass model, which is `x + A/3`. The `y` is
    /// unchanged.
    pub fn to_weierstrass_x(&self, x: &F::E) -> F::E {
        x.clone() + self.a.clone() * self.third()
    }

    fn third(&self) -> F::E {
        self.field
            .invert(&self.field.from_integer(3))
            .expect("three is a unit above characteristic three")
    }
}

impl<F> XPoint<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    /// The point `(x : z)`, which is `x/z` and infinity when `z` is zero.
    pub fn new(x: F::E, z: F::E) -> Self {
        XPoint { x, z }
    }

    pub fn x(&self) -> &F::E {
        &self.x
    }

    pub fn z(&self) -> &F::E {
        &self.z
    }
}

impl<F> Clone for XPoint<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    fn clone(&self) -> Self {
        XPoint {
            x: self.x.clone(),
            z: self.z.clone(),
        }
    }
}

/// Projective, so `(X : Z)` and `(cX : cZ)` are one point.
impl<F> PartialEq for XPoint<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        self.x.clone() * other.z.clone() == other.x.clone() * self.z.clone()
    }
}

impl<F> fmt::Display for XPoint<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({} : {})", self.x, self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finite_fields::FiniteField;
    use crate::integers::Integers;
    use crate::structures::{CommutativeMonoid, Domain, QuotientRing};

    type Fp = Arc<QuotientRing<Integers>>;

    /// Every nonsingular Montgomery curve over `F_p`.
    fn all_curves(f: &Fp) -> Vec<Arc<MontgomeryCurve<Fp>>> {
        let p: u32 = f.order().try_into().expect("a small field");
        (0..p)
            .filter_map(|a| MontgomeryCurve::new(f.clone(), f.element(a)))
            .collect()
    }

    #[test]
    fn the_weierstrass_model_is_the_same_curve() {
        for prime in [11u32, 13] {
            let f = Integers::modulo(prime);
            for n in 0..prime {
                let a = f.element(n);
                let singular = &a * &a == f.element(4);
                let Some(c) = MontgomeryCurve::new(f.clone(), a.clone()) else {
                    assert!(singular, "A = {a} was rejected without A^2 = 4");
                    continue;
                };
                assert!(!singular, "A = {a} has A^2 = 4 and was accepted");

                let w = c.to_weierstrass();
                assert_eq!(c.j_invariant(), w.j_invariant(), "j at A = {a}");
                // The substitution itself, rather than the coefficients it gives:
                // x + A/3 has the same y^2 over it as x has here.
                for m in 0..prime {
                    let x = f.element(m);
                    assert_eq!(
                        w.evaluate(&c.to_weierstrass_x(&x)),
                        c.evaluate(&x),
                        "x = {x} at A = {a}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_ladder_agrees_with_the_group_law() {
        let prime = 11u32;
        let f = Integers::modulo(prime);
        for c in all_curves(&f) {
            let a = c.coefficient();
            let w = c.to_weierstrass();
            let order: u32 = w.order().try_into().expect("a small curve");

            for n in 0..prime {
                let x = f.element(n);
                // Every x with a y, the points of order two among them.
                let Some(y) = f.sqrt(&c.evaluate(&x)) else {
                    continue;
                };
                let p = c.point(x.clone());
                let image = w
                    .point(c.to_weierstrass_x(&x), y)
                    .expect("the substitution keeps it on the curve");

                for k in 0..2 * order {
                    match w.multiply(k, &image).coordinates() {
                        None => assert!(c.is_infinity(&c.ladder(k, &p)), "[{k}]{x} at A = {a}"),
                        Some((target, _)) => {
                            let reached = c
                                .affine_x(&c.ladder(k, &p))
                                .expect("the group law did not reach infinity");
                            assert_eq!(
                                c.to_weierstrass_x(&reached),
                                *target,
                                "[{k}]{x} at A = {a}"
                            );
                        }
                    }
                }
            }
            assert!(
                c.is_infinity(&c.ladder(7, &c.infinity())),
                "[7] of infinity"
            );
        }
    }

    #[test]
    fn minus_a_is_the_quadratic_twist() {
        // -1 is a non-square exactly for p = 3 mod 4, and twisting by it sends
        // A to -A.
        for prime in [7u32, 11, 19, 23] {
            let f = Integers::modulo(prime);
            for c in all_curves(&f) {
                let a = c.coefficient().clone();
                let twist = MontgomeryCurve::new(f.clone(), f.zero() - a.clone())
                    .expect("-A squares to what A does");

                assert_eq!(c.j_invariant(), twist.j_invariant(), "j at A = {a}");
                assert_eq!(
                    c.to_weierstrass().order() + twist.to_weierstrass().order(),
                    BigInt::from(2 * prime + 2),
                    "the counts at A = {a}"
                );
            }
        }
    }

    #[test]
    fn an_x_off_the_curve_runs_on_the_twist() {
        // What x-only arithmetic buys: a non-square x^3 + Ax^2 + x is a point of the
        // twist, and the ladder there is the ladder on E_{-A} at -x.
        let prime = 19u32;
        let f = Integers::modulo(prime);
        for c in all_curves(&f) {
            let a = c.coefficient().clone();
            let twist = MontgomeryCurve::new(f.clone(), f.zero() - a.clone()).expect("the twist");

            for n in 0..prime {
                let x = f.element(n);
                if f.legendre_symbol(&c.evaluate(&x)) != -1 {
                    continue;
                }
                let mirrored = f.zero() - x.clone();
                for k in 0..2 * prime {
                    let here = c.ladder(k, &c.point(x.clone()));
                    let there = twist.ladder(k, &twist.point(mirrored.clone()));
                    match (c.affine_x(&here), twist.affine_x(&there)) {
                        (None, None) => {}
                        (Some(u), Some(v)) => {
                            assert_eq!(u, f.zero() - v, "[{k}]{x} at A = {a}");
                        }
                        _ => panic!("[{k}]{x} at A = {a}: one reached infinity, one did not"),
                    }
                }
            }
        }
    }

    #[test]
    fn the_csidh_curve_has_the_torsion_its_isogenies_want() {
        // p + 1 = 4 * 3 * 5 * 7, which is what makes 3, 5 and 7 isogenies available.
        let prime = 419u32;
        assert_eq!(prime + 1, 4 * 3 * 5 * 7);
        let f = Integers::modulo(prime);
        let c = MontgomeryCurve::new(f.clone(), f.zero()).expect("A = 0 is nonsingular");

        let w = c.to_weierstrass();
        assert_eq!(w.order(), BigInt::from(420), "E_0 has p + 1 points");
        assert!(w.is_supersingular());

        for l in [3u32, 5, 7] {
            let cofactor = 420 / l;
            let mut found = 0;
            for n in 1..40u32 {
                // Every x is on the curve or its twist, and both have 420 points, so
                // the cofactor multiple has order dividing l either way.
                let q = c.ladder(cofactor, &c.point(f.element(n)));
                if c.is_infinity(&q) {
                    continue;
                }
                assert!(c.is_infinity(&c.ladder(l, &q)), "l = {l} at x = {n}");
                found += 1;
            }
            assert!(
                found > 10,
                "only {found} of the x gave a point of order {l}"
            );
        }
    }
}
