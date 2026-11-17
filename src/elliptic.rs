//! Elliptic curves in short Weierstrass form, `y^2 = x^3 + ax + b`.
//!
//! The points form an [`AdditiveGroup`] under the chord-and-tangent law, with the
//! point at infinity as the identity. The base field must not have characteristic 2
//! or 3, which this form assumes.

use crate::finite_fields::FiniteField;
use crate::structures::{AdditiveGroup, CommutativeMonoid, Domain, Field, RingOps};
use num_bigint::{BigInt, Sign};
use std::fmt;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};
use std::sync::Arc;

impl<F> Curve<F>
where
    F: FiniteField + Clone,
    F::E: RingOps + Eq + Div<Output = F::E> + for<'x> MulAssign<&'x F::E>,
    i64: Mul<F::E, Output = F::E>,
{
    /// Whether the two curves are isomorphic over their own field, rather than only
    /// over its closure.
    ///
    /// Equal j-invariants make them isomorphic over the algebraic closure. Over `F_q`
    /// the isomorphisms are `(x, y) -> (u^2 x, u^3 y)`, carrying `(a, b)` to
    /// `(u^4 a, u^6 b)`, so one has to exist with `u` in the field itself. A curve with
    /// the same j-invariant but no such `u` is a twist.
    ///
    /// Both curves must be over the same field. Nothing enforces it: element equality
    /// reduces by this curve's field, so comparing across two gives an answer rather
    /// than an error. A debug assertion catches a differing order, which is as far as
    /// the types can see.
    ///
    /// # Panics
    /// If the characteristic is 2 or 3, where this form of the isomorphisms does not
    /// hold.
    pub fn is_isomorphic(&self, other: &Curve<F>) -> bool {
        let f = &self.field;
        debug_assert!(
            f.order() == other.field.order(),
            "the curves are over fields of different order"
        );
        assert!(
            !f.is_zero(&f.from_integer(6)),
            "the isomorphisms take this form only away from characteristic 2 and 3"
        );
        if self.j_invariant() != other.j_invariant() {
            return false;
        }

        // Equal j-invariants put both curves in the same case.
        let (a, b) = (&self.a, &self.b);
        let (c, d) = (&other.a, &other.b);
        if f.is_zero(a) {
            // j = 0, so both are y^2 = x^3 + b and u^6 = d/b.
            f.is_kth_power(&(d.clone() / b.clone()), 6)
        } else if f.is_zero(b) {
            // j = 1728, so both are y^2 = x^3 + ax and u^4 = c/a.
            f.is_kth_power(&(c.clone() / a.clone()), 4)
        } else {
            // Otherwise equal j forces a^3 d^2 = c^3 b^2, which makes this the only
            // candidate for u^2.
            let lambda = (a.clone() * d.clone()) / (c.clone() * b.clone());
            f.is_kth_power(&lambda, 2)
        }
    }
}

impl<F> Curve<F>
where
    F: FiniteField + Clone,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    /// The number of points, the point at infinity included.
    ///
    /// `x` contributes `1 + chi(x^3 + ax + b)` points for the quadratic character
    /// `chi`, so `#E(F_q) = q + 1 + sum_x chi(x^3 + ax + b)`. That is one
    /// exponentiation per field element, so it is for small fields.
    pub fn order(&self) -> BigInt {
        let field = &self.field;
        let q = field.order();
        let mut count = &q + 1u32;
        let mut index = BigInt::from(0u32);
        while index < q {
            let x = field.element_at(&index);
            count += field.legendre_symbol(&self.evaluate(&x));
            index += 1u32;
        }
        count
    }

    /// The trace of Frobenius, `q + 1 - #E(F_q)`, which Hasse bounds by `2*sqrt(q)`.
    pub fn trace_of_frobenius(&self) -> BigInt {
        self.field.order() + 1u32 - self.order()
    }
}

/// A point on a curve, or the point at infinity.
#[derive(Debug)]
pub struct Point<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    /// `None` is the point at infinity.
    coordinates: Option<(F::E, F::E)>,
    curve: Arc<Curve<F>>,
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
    pub fn new(field: F, a: F::E, b: F::E) -> Option<Arc<Self>>
    where
        F::E: for<'x> MulAssign<&'x F::E>,
        i64: Mul<F::E, Output = F::E>,
    {
        let curve = Curve { field, a, b };
        (!curve.field.is_zero(&curve.discriminant())).then(|| Arc::new(curve))
    }

    /// `-16(4a^3 + 27b^2)`, zero exactly when the curve is singular.
    pub fn discriminant(&self) -> F::E
    where
        F::E: for<'x> MulAssign<&'x F::E>,
        i64: Mul<F::E, Output = F::E>,
    {
        -16 * (4 * self.field.pow(&self.a, 3) + 27 * self.field.pow(&self.b, 2))
    }

    /// The j-invariant, `1728 * 4a^3 / (4a^3 + 27b^2)`.
    ///
    /// Two curves over the same field are isomorphic over its algebraic closure exactly
    /// when their j-invariants agree.
    pub fn j_invariant(&self) -> F::E
    where
        F::E: for<'x> MulAssign<&'x F::E> + Div<Output = F::E>,
        i64: Mul<F::E, Output = F::E>,
    {
        let four_a_cubed = 4 * self.field.pow(&self.a, 3);
        // Nonzero: `new` rejects the curves where it would vanish.
        1728 * four_a_cubed.clone() / (four_a_cubed + 27 * self.field.pow(&self.b, 2))
    }

    pub fn field(&self) -> &F {
        &self.field
    }

    /// The curve's coefficients `(a, b)`.
    pub fn coefficients(&self) -> (&F::E, &F::E) {
        (&self.a, &self.b)
    }

    /// `x^3 + ax + b`, the right-hand side of the curve equation.
    pub fn evaluate(&self, x: &F::E) -> F::E {
        x.clone() * x.clone() * x.clone() + self.a.clone() * x.clone() + self.b.clone()
    }

    /// Whether `(x, y)` satisfies the curve equation.
    pub fn contains(&self, x: &F::E, y: &F::E) -> bool {
        y.clone() * y.clone() == self.evaluate(x)
    }

    /// The point `(x, y)`, or [`None`] if it is not on the curve.
    pub fn point(self: &Arc<Self>, x: F::E, y: F::E) -> Option<Point<F>> {
        self.contains(&x, &y).then(|| Point {
            coordinates: Some((x, y)),
            curve: Arc::clone(self),
        })
    }

    /// The point at infinity, which is the group's identity.
    pub fn infinity(self: &Arc<Self>) -> Point<F> {
        Point {
            coordinates: None,
            curve: Arc::clone(self),
        }
    }

    /// `n * p`, by double-and-add. Negative `n` multiplies the negation.
    pub fn multiply<T: Into<BigInt>>(self: &Arc<Self>, n: T, p: &Point<F>) -> Point<F> {
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
    pub fn curve(&self) -> &Arc<Curve<F>> {
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
            curve: Arc::clone(&self.curve),
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
            Arc::ptr_eq(&self.curve, &other.curve),
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
            Arc::ptr_eq(&self.curve, &rhs.curve),
            "points belong to different curves"
        );
        let curve = Arc::clone(&self.curve);
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
            if field.is_zero(&(py.clone() + qy.clone())) {
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

impl<F> Domain for Arc<Curve<F>>
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

impl<F> CommutativeMonoid for Arc<Curve<F>>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    fn zero(&self) -> Self::E {
        self.infinity()
    }
}

impl<F> AdditiveGroup for Arc<Curve<F>>
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
    use crate::structures::{Monoid, QuotientRing};

    type Fp = Arc<QuotientRing<Integers>>;

    /// y^2 = x^3 + x + 6 over F_11, a standard worked example.
    fn curve() -> (Fp, Arc<Curve<Fp>>) {
        let f = Integers::modulo(11);
        let c = Curve::new(f.clone(), f.element(1), f.element(6)).expect("nonsingular");
        (f, c)
    }

    /// Every affine point, by brute force over the whole field.
    fn all_points(f: &Fp, c: &Arc<Curve<Fp>>) -> Vec<Point<Fp>> {
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

    /// Every nonsingular curve over `f`.
    fn all_curves(f: &Fp) -> Vec<Arc<Curve<Fp>>> {
        let p: u32 = f.order().try_into().unwrap();
        let mut curves = Vec::new();
        for a in 0..p {
            for b in 0..p {
                if let Some(c) = Curve::new(f.clone(), f.element(a), f.element(b)) {
                    curves.push(c);
                }
            }
        }
        curves
    }

    /// Whether some `u` in the field carries one curve's coefficients to the other's,
    /// found by trying every one. This is the definition `is_isomorphic` is meant to
    /// decide, worked out without any of its reasoning.
    fn some_u_carries(f: &Fp, from: &Curve<Fp>, to: &Curve<Fp>) -> bool {
        let p: u32 = f.order().try_into().unwrap();
        let (a, b) = from.coefficients();
        let (c, d) = to.coefficients();
        (1..p).any(|n| {
            let u = f.element(n);
            let square = &u * &u;
            let fourth = &square * &square;
            let sixth = &fourth * &square;
            &fourth * a == *c && &sixth * b == *d
        })
    }

    #[test]
    fn is_isomorphic_decides_whether_some_u_exists() {
        // Quartic in the prime, and 5 and 7 are already where the fourth and sixth
        // powers are proper subgroups of the squares.
        for prime in [5u32, 7] {
            let f = Integers::modulo(prime);
            let curves = all_curves(&f);
            for x in &curves {
                for y in &curves {
                    let (a, b) = x.coefficients();
                    let (c, d) = y.coefficients();
                    assert_eq!(
                        x.is_isomorphic(y),
                        some_u_carries(&f, x, y),
                        "F_{prime}: ({a}, {b}) against ({c}, {d})"
                    );
                }
            }
        }
    }

    /// The order found by trying every pair of coordinates, with no character sums.
    fn counted_by_hand<F>(c: &Curve<F>) -> BigInt
    where
        F: FiniteField + Clone,
        F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
    {
        let f = c.field();
        let q = f.order();
        let mut count = BigInt::from(1u32);
        let mut i = BigInt::from(0u32);
        while i < q {
            let x = f.element_at(&i);
            let mut j = BigInt::from(0u32);
            while j < q {
                if c.contains(&x, &f.element_at(&j)) {
                    count += 1u32;
                }
                j += 1u32;
            }
            i += 1u32;
        }
        count
    }

    #[test]
    fn counting_agrees_with_trying_every_pair() {
        for prime in [5u32, 7, 11, 13] {
            let f = Integers::modulo(prime);
            for c in all_curves(&f) {
                let (a, b) = c.coefficients();
                assert_eq!(
                    c.order(),
                    counted_by_hand(&c),
                    "F_{prime}: y^2 = x^3 + {a}x + {b}"
                );
            }
        }
    }

    #[test]
    fn counting_agrees_with_trying_every_pair_over_an_extension() {
        // F_25, where the field is not Z/q and x runs through element_at.
        let f = crate::finite_fields::finite_field(5, 2);
        for (a, b) in [(1u32, 1u32), (0, 2), (3, 0), (2, 4)] {
            let c = Curve::new(f.clone(), f.element_at(&a.into()), f.element_at(&b.into()))
                .expect("nonsingular");
            assert_eq!(c.order(), counted_by_hand(&c), "at ({a}, {b})");
        }
    }

    #[test]
    fn the_order_annihilates_every_point() {
        // Lagrange, which the counting argument knows nothing about.
        let f = Integers::modulo(11);
        assert_eq!(curve().1.order(), BigInt::from(13), "the worked example");
        for c in all_curves(&f) {
            let order = c.order();
            let (a, b) = c.coefficients();
            for p in all_points(&f, &c) {
                assert_eq!(
                    c.multiply(order.clone(), &p),
                    c.infinity(),
                    "{order} * {p} on y^2 = x^3 + {a}x + {b}"
                );
            }
        }
    }

    #[test]
    fn the_trace_stays_within_hasse_s_bound() {
        for prime in [5u32, 7, 11, 13, 17, 19] {
            let f = Integers::modulo(prime);
            for c in all_curves(&f) {
                let t = c.trace_of_frobenius();
                let (a, b) = c.coefficients();
                assert!(
                    &t * &t <= BigInt::from(4 * prime),
                    "F_{prime}: y^2 = x^3 + {a}x + {b} has t = {t}"
                );
            }
        }
    }

    #[test]
    fn a_quadratic_twist_has_the_opposite_trace() {
        // Scaling by (d^2, d^3) for a non-square d gives the twist, where x -> dx
        // sends the cubic to d^3 f(x). So x contributes 1 + chi(f(x)) points here and
        // 1 - chi(f(x)) there, two in all, and the counts add to 2q + 2.
        for prime in [7u32, 11, 13, 17] {
            let f = Integers::modulo(prime);
            let d = (2..prime)
                .map(|n| f.element(n))
                .find(|n| f.legendre_symbol(n) == -1)
                .expect("some non-square");
            let (d2, d3) = (f.pow(&d, 2), f.pow(&d, 3));
            for c in all_curves(&f) {
                let (a, b) = c.coefficients();
                let twist = Curve::new(f.clone(), d2.clone() * a.clone(), d3.clone() * b.clone())
                    .expect("scaling keeps the discriminant nonzero");
                assert_eq!(
                    c.trace_of_frobenius(),
                    -twist.trace_of_frobenius(),
                    "F_{prime}: y^2 = x^3 + {a}x + {b}"
                );
            }
        }
    }

    #[test]
    fn curves_sharing_a_j_invariant_fall_into_gcd_k_and_q_minus_one_classes() {
        use crate::number_theory::gcd;

        for prime in [5u32, 7, 11, 13, 17] {
            let f = Integers::modulo(prime);
            let curves = all_curves(&f);

            let mut seen: Vec<_> = Vec::new();
            for curve in &curves {
                let j = curve.j_invariant();
                if seen.contains(&j) {
                    continue;
                }
                seen.push(j.clone());

                let group: Vec<_> = curves.iter().filter(|c| c.j_invariant() == j).collect();
                // One representative per class, found by asking.
                let mut classes: Vec<&Arc<Curve<Fp>>> = Vec::new();
                for c in &group {
                    if !classes.iter().any(|r| r.is_isomorphic(c)) {
                        classes.push(c);
                    }
                }

                // The isomorphisms act through u^k, so the classes are F_q* modulo its
                // k-th powers: 6 at j = 0, 4 at j = 1728, 2 elsewhere, cut down by the
                // gcd when the field has no such roots.
                let (a, b) = group[0].coefficients();
                let k = if f.is_zero(a) {
                    6u32
                } else if f.is_zero(b) {
                    4
                } else {
                    2
                };
                let expected = gcd(BigInt::from(k), BigInt::from(prime - 1));
                assert_eq!(
                    BigInt::from(classes.len()),
                    expected,
                    "F_{prime}, j = {j}, k = {k}"
                );
            }
        }
    }

    #[test]
    fn the_quadratic_twist_shares_a_j_invariant_without_being_isomorphic() {
        let f = Integers::modulo(11);
        let base = Curve::new(f.clone(), f.element(1), f.element(6)).expect("nonsingular");

        // (a, b) -> (u^4 a, u^6 b) is an isomorphism for every u.
        for u in 1..11i64 {
            let (u4, u6) = (u.pow(4) % 11, u.pow(6) % 11);
            let same =
                Curve::new(f.clone(), f.element(u4), f.element(6 * u6 % 11)).expect("nonsingular");
            assert!(base.is_isomorphic(&same), "at u = {u}");
        }

        // Scaling by (d^2, d^3) for a non-square d keeps j but is a twist: there is no
        // u with u^2 = d.
        let two = f.element(2);
        assert_eq!(f.legendre_symbol(&two), -1, "2 is a non-square mod 11");
        let twist =
            Curve::new(f.clone(), f.element(4), f.element(6 * 8 % 11)).expect("nonsingular");
        assert_eq!(twist.j_invariant(), base.j_invariant());
        assert!(!base.is_isomorphic(&twist));
    }

    #[test]
    fn the_j_invariant_is_an_isomorphism_invariant() {
        let f = Integers::modulo(11);
        let curve = |a: i64, b: i64| Curve::new(f.clone(), f.element(a), f.element(b));

        // (a, b) -> (a u^4, b u^6) is the substitution x -> u^2 x, y -> u^3 y, so those
        // curves are isomorphic and must agree.
        let base = curve(1, 6).expect("nonsingular");
        for u in 1..11i64 {
            let (u4, u6) = (u.pow(4) % 11, u.pow(6) % 11);
            let twisted = curve(u4, 6 * u6 % 11).expect("nonsingular");
            assert_eq!(twisted.j_invariant(), base.j_invariant(), "at u = {u}");
        }

        // b = 0 leaves 1728 * 4a^3 / 4a^3, and a = 0 leaves a zero numerator.
        assert_eq!(curve(1, 0).unwrap().j_invariant(), 1728 * f.identity());
        assert_eq!(curve(0, 1).unwrap().j_invariant(), f.zero());

        // Against the discriminant, which it is defined in terms of: j * disc is
        // -1728 * (4a)^3.
        for (a, b) in [(1i64, 6i64), (3, 2), (5, 9), (1, 0), (0, 7)] {
            let c = curve(a, b).expect("nonsingular");
            let four_a = 4 * f.element(a);
            assert_eq!(
                c.j_invariant() * c.discriminant(),
                -1728 * f.pow(&four_a, 3),
                "at (a, b) = ({a}, {b})"
            );
        }

        // Curves that are not isomorphic are told apart.
        assert_ne!(
            curve(1, 6).unwrap().j_invariant(),
            curve(0, 1).unwrap().j_invariant()
        );
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
