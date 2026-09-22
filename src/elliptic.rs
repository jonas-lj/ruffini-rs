//! Elliptic curves in short Weierstrass form, `y^2 = x^3 + ax + b`.
//!
//! The points form an [`AdditiveGroup`] under the chord-and-tangent law, with the
//! point at infinity as the identity. The base field must not have characteristic 2
//! or 3, which this form assumes.
//!
//! No pairings. They are the part of the Java library that was never audited, they
//! need extension field towers this crate does not build yet, and getting one subtly
//! wrong is not the kind of bug tests here would catch.

use crate::pow::pow;
use crate::structures::{AdditiveGroup, CommutativeMonoid, Domain, Field, RingOps};
use std::ops::MulAssign;
use num_bigint::{BigInt, Sign};
use std::fmt;
use std::ops::{Add, AddAssign, Neg, Sub};
use std::rc::Rc;

/// A point on a curve, or the point at infinity.
#[derive(Debug)]
pub struct Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    /// `None` is the point at infinity.
    coordinates: Option<(F::E, F::E)>,
    curve: Rc<Curve<F>>,
}

/// The curve `y^2 = x^3 + ax + b` over a field.
#[derive(Debug)]
pub struct Curve<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    field: F,
    a: F::E,
    b: F::E,
}

impl<F> Curve<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    /// The curve `y^2 = x^3 + ax + b`, or [`None`] if it is singular.
    pub fn new(field: F, a: F::E, b: F::E) -> Option<Rc<Self>>
    where
        F::E: for<'x> MulAssign<&'x F::E>,
    {
        let curve = Curve { field, a, b };
        (curve.discriminant() != curve.field.zero()).then(|| Rc::new(curve))
    }

    /// `-16(4a^3 + 27b^2)`, zero exactly when the curve is singular.
    pub fn discriminant(&self) -> F::E
    where
        F::E: for<'x> MulAssign<&'x F::E>,
    {
        let f = &self.field;
        let inner =
            f.from_integer(4) * pow(f, &self.a, 3) + f.from_integer(27) * pow(f, &self.b, 2);
        f.zero() - f.from_integer(16) * inner
    }

    pub fn field(&self) -> &F {
        &self.field
    }

    /// The curve's coefficients `(a, b)`.
    pub fn coefficients(&self) -> (&F::E, &F::E) {
        (&self.a, &self.b)
    }

    /// Whether `(x, y)` satisfies the curve equation.
    pub fn contains(&self, x: &F::E, y: &F::E) -> bool {
        let lhs = y.clone() * y.clone();
        let rhs = x.clone() * x.clone() * x.clone() + self.a.clone() * x.clone() + self.b.clone();
        lhs == rhs
    }

    /// The point `(x, y)`, or [`None`] if it is not on the curve.
    pub fn point(self: &Rc<Self>, x: F::E, y: F::E) -> Option<Point<F>> {
        self.contains(&x, &y).then(|| Point {
            coordinates: Some((x, y)),
            curve: Rc::clone(self),
        })
    }

    /// The point at infinity, which is the group's identity.
    pub fn infinity(self: &Rc<Self>) -> Point<F> {
        Point {
            coordinates: None,
            curve: Rc::clone(self),
        }
    }

    /// `n * p`, by double-and-add. Negative `n` multiplies the negation.
    pub fn multiply<T: Into<BigInt>>(self: &Rc<Self>, n: T, p: &Point<F>) -> Point<F> {
        let (sign, magnitude) = n.into().into_parts();
        let mut result = self.infinity();
        let mut addend = p.clone();
        let bits = magnitude.bits();
        for i in 0..bits {
            if magnitude.bit(i) {
                result += addend.clone();
            }
            if i + 1 < bits {
                addend = addend.clone() + addend;
            }
        }
        match sign {
            Sign::Minus => -result,
            _ => result,
        }
    }
}

impl<F> Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    /// The affine coordinates, or [`None`] at infinity.
    pub fn coordinates(&self) -> Option<(&F::E, &F::E)> {
        self.coordinates.as_ref().map(|(x, y)| (x, y))
    }

    pub fn is_infinity(&self) -> bool {
        self.coordinates.is_none()
    }

    /// The curve this point lies on.
    pub fn curve(&self) -> &Rc<Curve<F>> {
        &self.curve
    }
}

impl<F> Clone for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    fn clone(&self) -> Self {
        Point {
            coordinates: self.coordinates.clone(),
            curve: Rc::clone(&self.curve),
        }
    }
}

impl<F> PartialEq for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        debug_assert!(
            Rc::ptr_eq(&self.curve, &other.curve),
            "points belong to different curves"
        );
        self.coordinates == other.coordinates
    }
}

impl<F> Eq for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
}

impl<F> Neg for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        let zero = self.curve.field.zero();
        Point {
            coordinates: self.coordinates.map(|(x, y)| (x, zero - y)),
            curve: self.curve,
        }
    }
}

/// The chord-and-tangent law.
impl<F> Add for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        debug_assert!(
            Rc::ptr_eq(&self.curve, &rhs.curve),
            "points belong to different curves"
        );
        let curve = Rc::clone(&self.curve);
        let field = &curve.field;

        let (px, py) = match self.coordinates {
            None => return rhs,
            Some(p) => p,
        };
        let (qx, qy) = match rhs.coordinates {
            None => {
                return Point {
                    coordinates: Some((px, py)),
                    curve,
                }
            }
            Some(q) => q,
        };

        let slope = if px == qx {
            // Vertical line: the points are inverse, so the sum is the identity. This
            // also covers doubling a point of order two, where y is zero.
            if py.clone() + qy.clone() == field.zero() {
                return curve.infinity();
            }
            // Tangent at p. Needs characteristic not 2, since it divides by 2y.
            let numerator = field.from_integer(3) * px.clone() * px.clone() + curve.a.clone();
            let denominator = field.from_integer(2) * py.clone();
            numerator
                * field
                    .invert(&denominator)
                    .expect("2y is nonzero away from the vertical case")
        } else {
            let numerator = qy - py.clone();
            let denominator = qx.clone() - px.clone();
            numerator
                * field
                    .invert(&denominator)
                    .expect("x difference is nonzero here")
        };

        let rx = slope.clone() * slope.clone() - px.clone() - qx;
        let ry = slope * (px - rx.clone()) - py;
        Point {
            coordinates: Some((rx, ry)),
            curve,
        }
    }
}

impl<F> Sub for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<F> AddAssign for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    fn add_assign(&mut self, rhs: Self) {
        let placeholder = self.curve.infinity();
        let lhs = std::mem::replace(self, placeholder);
        *self = lhs + rhs;
    }
}

impl<F> Domain for Rc<Curve<F>>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    type E = Point<F>;
    /// `None` is the point at infinity.
    type Repr = Option<(F::E, F::E)>;

    /// # Panics
    /// If the coordinates are not on the curve. Use [`Curve::point`] to check.
    fn element<T: Into<Self::Repr>>(&self, value: T) -> Self::E {
        match value.into() {
            None => self.infinity(),
            Some((x, y)) => self.point(x, y).expect("point is not on the curve"),
        }
    }
}

impl<F> CommutativeMonoid for Rc<Curve<F>>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    fn zero(&self) -> Self::E {
        self.infinity()
    }
}

impl<F> AdditiveGroup for Rc<Curve<F>>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
}

impl<F> fmt::Display for Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.coordinates {
            None => write!(f, "O"),
            Some((x, y)) => write!(f, "({x}, {y})"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::Integers;
    use crate::structures::QuotientRing;

    type Fp = Rc<QuotientRing<Integers>>;

    /// y^2 = x^3 + x + 6 over F_11, a standard worked example.
    fn curve() -> (Fp, Rc<Curve<Fp>>) {
        let f = Integers::modulo(11);
        let c = Curve::new(f.clone(), f.element(1), f.element(6)).expect("nonsingular");
        (f, c)
    }

    /// Every affine point, by brute force over the whole field.
    fn all_points(f: &Fp, c: &Rc<Curve<Fp>>) -> Vec<Point<Fp>> {
        let mut points = vec![c.infinity()];
        for x in 0..11i64 {
            for y in 0..11i64 {
                if let Some(p) = c.point(f.element(x), f.element(y)) {
                    points.push(p);
                }
            }
        }
        points
    }

    #[test]
    fn the_points_form_a_group() {
        let (f, c) = curve();
        let points = all_points(&f, &c);

        // This curve has 13 points including infinity, so the group is cyclic of
        // prime order and every affine point generates it.
        assert_eq!(points.len(), 13);

        for p in &points {
            // Identity, and inverses.
            assert_eq!(p.clone() + c.infinity(), *p);
            assert_eq!(c.infinity() + p.clone(), *p);
            assert_eq!(p.clone() + (-p.clone()), c.infinity());
            assert_eq!(p.clone() - p.clone(), c.infinity());

            for q in &points {
                let sum = p.clone() + q.clone();
                // Closure: the sum is on the curve.
                if let Some((x, y)) = sum.coordinates() {
                    assert!(c.contains(x, y), "{p} + {q} = {sum} is off the curve");
                }
                // Commutativity.
                assert_eq!(sum, q.clone() + p.clone());
            }
        }

        // Associativity, over every triple.
        for p in &points {
            for q in &points {
                for r in &points {
                    assert_eq!(
                        (p.clone() + q.clone()) + r.clone(),
                        p.clone() + (q.clone() + r.clone()),
                        "associativity at {p}, {q}, {r}"
                    );
                }
            }
        }
    }

    #[test]
    fn doubling_and_scalar_multiplication() {
        let (f, c) = curve();
        let g = c.point(f.element(2), f.element(7)).expect("on the curve");

        // Doubling by hand: s = (3*4 + 1)/(2*7) = 13/14 = 2/3 = 2*4 = 8 in F_11,
        // so x = 64 - 4 = 60 = 5 and y = 8*(2 - 5) - 7 = -31 = 2.
        assert_eq!(
            g.clone() + g.clone(),
            c.point(f.element(5), f.element(2)).unwrap()
        );
        assert_eq!(c.multiply(2, &g), g.clone() + g.clone());

        // Scalar multiplication agrees with repeated addition, and wraps at the order.
        let mut running = c.infinity();
        for n in 0..15i64 {
            assert_eq!(c.multiply(n, &g), running, "at n = {n}");
            running += g.clone();
        }
        assert_eq!(c.multiply(13, &g), c.infinity(), "13 is the group order");
        assert_eq!(c.multiply(0, &g), c.infinity());

        // Negative scalars negate.
        for n in 1..13i64 {
            assert_eq!(c.multiply(-n, &g), -c.multiply(n, &g));
            assert_eq!(c.multiply(n, &g) + c.multiply(-n, &g), c.infinity());
        }

        // Large scalars reduce modulo the order.
        assert_eq!(c.multiply(13 * 1000 + 4, &g), c.multiply(4, &g));
    }

    #[test]
    fn points_of_order_two_double_to_infinity() {
        // y^2 = x^3 - x over F_11 has three points with y = 0, each of order two.
        let f = Integers::modulo(11);
        let c = Curve::new(f.clone(), f.element(-1), f.element(0)).expect("nonsingular");
        for x in [0i64, 1, 10] {
            let p = c.point(f.element(x), f.element(0)).expect("on the curve");
            assert_eq!(p.clone() + p.clone(), c.infinity(), "2 * {p}");
            assert_eq!(c.multiply(2, &p), c.infinity());
            assert_eq!(-p.clone(), p);
        }
    }

    #[test]
    fn rejects_singular_curves_and_points_off_the_curve() {
        let f = Integers::modulo(11);
        // 4a^3 + 27b^2 = 0 with a = b = 0 is the cusp y^2 = x^3.
        assert!(Curve::new(f.clone(), f.element(0), f.element(0)).is_none());

        let (f, c) = curve();
        assert!(c.point(f.element(2), f.element(7)).is_some());
        assert!(c.point(f.element(2), f.element(6)).is_none());
        assert_eq!(c.discriminant(), f.element(-16 * (4 + 27 * 36)));
    }

    #[test]
    #[should_panic(expected = "not on the curve")]
    fn element_rejects_a_point_off_the_curve() {
        let (f, c) = curve();
        c.element(Some((f.element(2), f.element(6))));
    }

    #[test]
    fn display_and_the_domain_surface() {
        let (f, c) = curve();
        assert_eq!(format!("{}", c.infinity()), "O");
        assert_eq!(format!("{}", c.point(f.element(2), f.element(7)).unwrap()), "(2, 7)");
        // zero() is the point at infinity.
        assert_eq!(c.zero(), c.infinity());
        assert_eq!(c.element(None), c.infinity());
        assert_eq!(
            c.element(Some((f.element(2), f.element(7)))),
            c.point(f.element(2), f.element(7)).unwrap()
        );
    }
}
