//! Elliptic curves in short Weierstrass form, `y^2 = x^3 + ax + b`.
//!
//! The points form an [`AdditiveGroup`] under the chord-and-tangent law, with the
//! point at infinity as the identity. The base field must not have characteristic 2
//! or 3, which this form assumes.

use crate::finite_fields::FiniteField;
use crate::fractions::Fraction;
use crate::polynomials::{Polynomial, PolynomialRing, RingExt};
use crate::structures::{
    AdditiveGroup, CommutativeMonoid, Domain, EuclideanDomain, Field, Monoid, RingOps,
};
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

    /// One isomorphism onto `other` over the field itself, or [`None`] when there is
    /// none. [`Curve::isomorphisms`] gives them all.
    ///
    /// # Panics
    /// If the characteristic is 2 or 3, as [`Curve::is_isomorphic`] does.
    pub fn isomorphism(self: &Arc<Self>, other: &Arc<Curve<F>>) -> Option<Isomorphism<F>> {
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
            return None;
        }

        // The same three cases as `is_isomorphic`, asking for the root rather than
        // whether there is one.
        let (a, b) = (&self.a, &self.b);
        let (c, d) = (&other.a, &other.b);
        let u = if f.is_zero(a) {
            sixth_root(f, &(d.clone() / b.clone()))?
        } else if f.is_zero(b) {
            fourth_root(f, &(c.clone() / a.clone()))?
        } else {
            f.sqrt(&((a.clone() * d.clone()) / (c.clone() * b.clone())))?
        };
        Some(Isomorphism {
            domain: Arc::clone(self),
            codomain: Arc::clone(other),
            u,
        })
    }

    /// Every isomorphism onto `other` over the field itself, which is one of them
    /// times each root of unity in `mu_k`, and so empty when there is none.
    ///
    /// # Panics
    /// If the characteristic is 2 or 3, as [`Curve::is_isomorphic`] does.
    pub fn isomorphisms(self: &Arc<Self>, other: &Arc<Curve<F>>) -> Vec<Isomorphism<F>> {
        let Some(first) = self.isomorphism(other) else {
            return Vec::new();
        };
        roots_of_unity(&self.field, self.automorphism_degree())
            .into_iter()
            .map(|zeta| Isomorphism {
                domain: Arc::clone(self),
                codomain: Arc::clone(other),
                u: first.u.clone() * zeta,
            })
            .collect()
    }

    /// The `k` with `Aut(E) = mu_k`: 6 where `j` is 0, 4 where it is 1728, and 2
    /// everywhere else.
    fn automorphism_degree(&self) -> u32 {
        if self.field.is_zero(&self.a) {
            6
        } else if self.field.is_zero(&self.b) {
            4
        } else {
            2
        }
    }
}

/// A fourth root of `v`, or [`None`] when it has none.
fn fourth_root<F>(field: &F, v: &F::E) -> Option<F::E>
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    let square = field.sqrt(v)?;
    // Both signs, because which of the two square roots `sqrt` hands back is not
    // something it promises, and where -1 is not a square only one of them is.
    field
        .sqrt(&square)
        .or_else(|| field.sqrt(&(field.zero() - square.clone())))
}

/// A sixth root of `v`, or [`None`] when it has none.
fn sixth_root<F>(field: &F, v: &F::E) -> Option<F::E>
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    // The cube root first: the cube roots of a sixth power differ by an element of
    // order three, which is a square like every element of odd order, so either cube
    // root will do and there is no case to split.
    field.sqrt(&field.root(v, 3)?)
}

/// The `k`-th roots of unity, for `k` of 2, 4 or 6: `mu_k` is cyclic of order
/// `gcd(k, q - 1)`, which away from characteristic two is 2, 4 or 6 again.
fn roots_of_unity<F>(field: &F, k: u32) -> Vec<F::E>
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    let one = field.identity();
    let order = crate::number_theory::gcd(k, field.order() - 1u32);
    let order: u32 = order.try_into().expect("a divisor of 6");
    let generator = match order {
        2 => field.zero() - one.clone(),
        4 => field
            .sqrt(&(field.zero() - one.clone()))
            .expect("4 divides q - 1, so -1 is a square"),
        6 => field.zero() - field.root_of_unity(3).expect("3 divides q - 1"),
        _ => panic!("mu_{k} has {order} elements, which no odd order allows"),
    };
    (0..order).map(|j| field.pow(&generator, j)).collect()
}

/// An isomorphism `(x, y) -> (u^2 x, u^3 y)`, which away from characteristic 2 and 3
/// is the only shape one can take between curves in this form.
#[derive(Debug)]
pub struct Isomorphism<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    domain: Arc<Curve<F>>,
    codomain: Arc<Curve<F>>,
    u: F::E,
}

impl<F> Isomorphism<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    pub fn domain(&self) -> &Arc<Curve<F>> {
        &self.domain
    }

    pub fn codomain(&self) -> &Arc<Curve<F>> {
        &self.codomain
    }

    /// The `u` of `(x, y) -> (u^2 x, u^3 y)`, which carries `(a, b)` to
    /// `(u^4 a, u^6 b)`.
    pub fn u(&self) -> &F::E {
        &self.u
    }

    /// The image of `p`, with infinity fixed.
    pub fn evaluate(&self, p: &Point<F>) -> Point<F> {
        let Some((x, y)) = p.coordinates() else {
            return self.codomain.infinity();
        };
        let square = self.u.clone() * self.u.clone();
        let cube = square.clone() * self.u.clone();
        self.codomain
            .point(square * x.clone(), cube * y.clone())
            .expect("an isomorphism lands on the codomain")
    }

    /// The isomorphism back, which is the one with `u` inverted.
    pub fn inverse(&self) -> Isomorphism<F> {
        Isomorphism {
            domain: Arc::clone(&self.codomain),
            codomain: Arc::clone(&self.domain),
            u: self
                .domain
                .field
                .invert(&self.u)
                .expect("u is nonzero, being a root of a nonzero coefficient ratio"),
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

    /// Whether the curve is supersingular, that is whether the characteristic divides
    /// the trace. Over `F_p` Hasse's bound leaves only `t = 0`, but over an extension
    /// the trace can be a nonzero multiple of `p`.
    pub fn is_supersingular(&self) -> bool {
        self.trace_of_frobenius() % self.field.characteristic() == BigInt::from(0u32)
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
    polynomials: Arc<PolynomialRing<F>>,
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
        let polynomials = field.polynomials();
        let curve = Curve {
            field,
            a,
            b,
            polynomials,
        };
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

    /// `F[x]`, the ring the curve's polynomials live in.
    pub fn polynomials(&self) -> &Arc<PolynomialRing<F>> {
        &self.polynomials
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

    /// The division polynomials `g_0` up to `g_n`, where `psi_n` is `g_n` for odd `n`
    /// and `y * g_n` for even `n`, which puts every one of them in `F[x]`.
    ///
    /// Each is built from its predecessors, so this is what asking for one costs.
    pub fn division_polynomials(&self, n: usize) -> Vec<Polynomial<F>> {
        let ring = &self.polynomials;
        let x = ring.indeterminate();
        let a = ring.constant(self.a.clone());
        let b = ring.constant(self.b.clone());
        let f_squared = self.cubic() * self.cubic();

        let mut g = vec![
            ring.zero(),
            ring.identity(),
            2 * ring.identity(),
            3 * x.pow(4) + 6 * (a.clone() * x.pow(2)) + 12 * (b.clone() * x.clone())
                - a.clone() * a.clone(),
            4 * (x.pow(6) + 5 * (a.clone() * x.pow(4)) + 20 * (b.clone() * x.pow(3))
                - 5 * (a.clone() * a.clone() * x.pow(2))
                - 4 * (a.clone() * b.clone() * x.clone())
                - 8 * (b.clone() * b.clone())
                - a.pow(3)),
        ];
        g.truncate(n + 1);

        let half = ring.constant(
            self.field
                .invert(&self.field.from_integer(2))
                .expect("the curve form rules out characteristic two"),
        );
        for k in 5..=n {
            let m = k / 2;
            let next = if k % 2 == 1 {
                let left = g[m + 2].clone() * g[m].pow(3);
                let right = g[m - 1].clone() * g[m + 1].pow(3);
                // y^4 = f^2 lands on whichever side holds the two even indices.
                if m % 2 == 0 {
                    f_squared.clone() * left - right
                } else {
                    left - f_squared.clone() * right
                }
            } else {
                let inner = g[m + 2].clone() * g[m - 1].pow(2) - g[m - 2].clone() * g[m + 1].pow(2);
                half.clone() * g[m].clone() * inner
            };
            g.push(next);
        }
        g
    }

    /// The `n`-th division polynomial `g_n`, where `psi_n` is `g_n` for odd `n` and
    /// `y * g_n` for even `n`.
    pub fn division_polynomial(&self, n: usize) -> Polynomial<F> {
        self.division_polynomials(n)
            .pop()
            .expect("the list runs from g_0")
    }

    /// Multiplication by `n` on the `x` coordinate, as `(phi_n, psi_n^2)` with
    /// `x([n]P) = phi_n(x) / psi_n^2(x)`. Both are in `F[x]`, and `psi_n^2` vanishes
    /// exactly at the `x` of the points `[n]` sends to infinity.
    ///
    /// # Panics
    /// If `n` is zero, where `[0]P` is infinity and has no `x`.
    pub fn multiplication_map(&self, n: usize) -> (Polynomial<F>, Polynomial<F>) {
        assert!(n > 0, "[0]P is the point at infinity, which has no x");
        let f = self.cubic();
        let g = self.division_polynomials(n + 1);

        // Exactly one of the two products carries an f, n and n +- 1 being of
        // opposite parity.
        let (psi_squared, cross) = if n % 2 == 1 {
            (
                g[n].clone() * g[n].clone(),
                f * g[n + 1].clone() * g[n - 1].clone(),
            )
        } else {
            (
                f * g[n].clone() * g[n].clone(),
                g[n + 1].clone() * g[n - 1].clone(),
            )
        };
        (
            self.polynomials.indeterminate() * psi_squared.clone() - cross,
            psi_squared,
        )
    }

    /// `x^3 + ax + b`, the right-hand side of the curve equation, in `F[x]`.
    pub fn cubic(&self) -> Polynomial<F> {
        let ring = &self.polynomials;
        let x = ring.indeterminate();
        x.pow(3) + ring.constant(self.a.clone()) * x + ring.constant(self.b.clone())
    }

    /// The separable isogeny with kernel `<generator>`, by Vélu's formulas, taking
    /// `order` to be the order of `generator`. The point at infinity, of order one,
    /// gives the identity map.
    ///
    /// # Panics
    /// If `order` is not the order of `generator`, which walking the kernel settles
    /// either way: no multiple before the last may be infinity, and the last must be.
    pub fn isogeny(self: &Arc<Self>, generator: &Point<F>, order: usize) -> Isogeny<F>
    where
        F::E: for<'x> MulAssign<&'x F::E>,
        i64: Mul<F::E, Output = F::E>,
    {
        let field = &self.field;

        // One Q from each pair {Q, -Q}, which the x coordinate tells apart, together
        // with the points of order two, each of which is its own pair.
        let mut summands: Vec<(F::E, F::E, F::E)> = Vec::new();
        let mut q = generator.clone();
        for _ in 1..order {
            let (x, y) = q
                .coordinates()
                .map(|(x, y)| (x.clone(), y.clone()))
                .expect("the generator's order is smaller than the order given");
            q += generator.clone();

            if summands.iter().any(|(seen, _, _)| *seen == x) {
                continue;
            }
            let g = field.from_integer(3) * x.clone() * x.clone() + self.a.clone();
            let v = if field.is_zero(&y) {
                g
            } else {
                field.from_integer(2) * g
            };
            let u = field.from_integer(4) * y.clone() * y;
            summands.push((x, u, v));
        }
        // The loop advanced order - 1 times from the generator, so this is order * P.
        assert!(
            order > 0 && q.is_infinity(),
            "the generator's order is larger than {order}"
        );

        let mut v_sum = field.zero();
        let mut w_sum = field.zero();
        for (x, u, v) in &summands {
            v_sum += v.clone();
            w_sum += u.clone() + x.clone() * v.clone();
        }
        let codomain = Curve::new(
            field.clone(),
            self.a.clone() - 5 * v_sum,
            self.b.clone() - 7 * w_sum,
        )
        .expect("the codomain of an isogeny is nonsingular");

        Isogeny {
            domain: Arc::clone(self),
            codomain,
            degree: order,
            summands,
        }
    }
}

/// A separable isogeny: a homomorphism onto another curve, of the shape
/// `(x, y) -> (r1(x), y * r2(x))`, whose kernel is the finite subgroup it sends to
/// infinity.
///
/// Neither rational function is stored, only the summands they are both built from,
/// one for each point of Vélu's `S`.
#[derive(Debug)]
pub struct Isogeny<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    domain: Arc<Curve<F>>,
    codomain: Arc<Curve<F>>,
    degree: usize,
    summands: Vec<(F::E, F::E, F::E)>,
}

impl<F> Isogeny<F>
where
    F: Field + Clone,
    F::E: RingOps + Eq,
{
    pub fn domain(&self) -> &Arc<Curve<F>> {
        &self.domain
    }

    /// The curve the isogeny maps onto, as a handle of its own even when it has the
    /// same coefficients as the domain.
    pub fn codomain(&self) -> &Arc<Curve<F>> {
        &self.codomain
    }

    /// The degree, which is the order of the kernel.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Vélu's `(x_Q, u_Q, v_Q)` for each `Q` in `S`, which holds one of each pair
    /// `{Q, -Q}` in the kernel: `u_Q` is `4y_Q^2`, and `v_Q` is `3x_Q^2 + a`, doubled
    /// away from the 2-torsion.
    pub fn summands(&self) -> &[(F::E, F::E, F::E)] {
        &self.summands
    }

    /// The image of `p`, which is infinity exactly on the kernel, from
    /// `r1 = x + sum [v_Q/(x - x_Q) + u_Q/(x - x_Q)^2]` and its derivative
    /// `r2 = 1 - sum [v_Q/(x - x_Q)^2 + 2 u_Q/(x - x_Q)^3]`.
    pub fn evaluate(&self, p: &Point<F>) -> Point<F> {
        let field = &self.domain.field;
        let Some((x, y)) = p.coordinates() else {
            return self.codomain.infinity();
        };
        let two = field.from_integer(2);

        let mut r1 = x.clone();
        let mut r2 = field.identity();
        for (x_q, u, v) in &self.summands {
            let Some(inverse) = field.invert(&(x.clone() - x_q.clone())) else {
                // The kernel is exactly where one of the summands has a pole.
                return self.codomain.infinity();
            };
            let square = inverse.clone() * inverse.clone();
            let cube = square.clone() * inverse.clone();
            r1 = r1 + v.clone() * inverse + u.clone() * square.clone();
            r2 = r2 - v.clone() * square - two.clone() * u.clone() * cube;
        }

        self.codomain
            .point(r1, y.clone() * r2)
            .expect("Vélu's formulas land on the codomain")
    }

    /// `r1` as an element of `F(x)`, in lowest terms, so that its denominator is of
    /// degree `n - 1` and vanishes exactly on the kernel.
    ///
    /// The `y` coordinate needs no second map: `r2` is the derivative of this.
    pub fn x_map(&self) -> Fraction<Arc<PolynomialRing<F>>> {
        let ring = self.domain.polynomials();
        let functions = ring.fractions();
        let x = ring.indeterminate();

        let mut r1 = functions.fraction(x.clone(), ring.identity());
        for (x_q, u, v) in &self.summands {
            let d = x.clone() - ring.constant(x_q.clone());
            r1 = r1
                + functions.fraction(ring.constant(v.clone()), d.clone())
                + functions.fraction(ring.constant(u.clone()), d.clone() * d);
        }
        r1
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
    use crate::structures::{Monoid, QuotientRing, Ring};

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

    #[test]
    fn fourth_and_sixth_roots_exist_exactly_for_the_powers() {
        for prime in [5u32, 7, 11, 13, 17, 19, 37] {
            let f = Integers::modulo(prime);
            let all: Vec<_> = (0..prime as i64).map(|n| f.element(n)).collect();
            let fourth: Vec<_> = all.iter().map(|x| f.pow(x, 4)).collect();
            let sixth: Vec<_> = all.iter().map(|x| f.pow(x, 6)).collect();

            for x in &all {
                match fourth_root(&f, x) {
                    Some(r) => assert_eq!(f.pow(&r, 4), *x, "F_{prime}"),
                    None => assert!(!fourth.contains(x), "F_{prime}: {x} is a fourth power"),
                }
                match sixth_root(&f, x) {
                    Some(r) => assert_eq!(f.pow(&r, 6), *x, "F_{prime}"),
                    None => assert!(!sixth.contains(x), "F_{prime}: {x} is a sixth power"),
                }
            }
        }
    }

    #[test]
    fn an_isomorphism_exists_exactly_where_is_isomorphic_says_and_carries_the_pair() {
        // 13 = 1 mod 12, so F_13 has all of mu_4 and mu_6 and every branch runs.
        for prime in [7u32, 11, 13] {
            let f = Integers::modulo(prime);
            let curves = all_curves(&f);
            // Isomorphism is symmetric, and `inverse` carries the other direction, so
            // the unordered pairs are enough.
            for (i, x) in curves.iter().enumerate() {
                for y in curves.iter().skip(i) {
                    let (a, b) = x.coefficients();
                    let (c, d) = y.coefficients();
                    let label = format!("F_{prime}: ({a}, {b}) to ({c}, {d})");
                    match x.isomorphism(y) {
                        Some(iso) => {
                            assert!(x.is_isomorphic(y), "{label}");
                            let u = iso.u();
                            assert_eq!(f.pow(u, 4) * a.clone(), *c, "u^4 a, {label}");
                            assert_eq!(f.pow(u, 6) * b.clone(), *d, "u^6 b, {label}");
                        }
                        None => assert!(!x.is_isomorphic(y), "{label}"),
                    }
                }
            }
        }
    }

    #[test]
    fn the_isomorphism_carries_the_group_across() {
        // u = 2 takes (1, 6) to (2^4, 2^6 * 6) = (5, 10) over F_11.
        let (f, c) = curve();
        let other = Curve::new(f.clone(), f.element(5), f.element(10)).expect("nonsingular");
        let iso = c
            .isomorphism(&other)
            .expect("u = 2 carries one to the other");

        let points = all_points(&f, &c);
        let images: Vec<Point<Fp>> = points.iter().map(|p| iso.evaluate(p)).collect();

        // Onto E'(F_11), and one to one.
        let targets = all_points(&f, &other);
        assert_eq!(images.len(), targets.len(), "13 points either side");
        for (i, image) in images.iter().enumerate() {
            assert!(targets.contains(image), "{} is not on the codomain", image);
            assert!(
                !images[i + 1..].contains(image),
                "two points share an image"
            );
        }

        // And additive, over every pair.
        for (p, image) in points.iter().zip(&images) {
            for (q, other_image) in points.iter().zip(&images) {
                assert_eq!(
                    iso.evaluate(&(p.clone() + q.clone())),
                    image.clone() + other_image.clone(),
                    "additivity at {p}, {q}"
                );
            }
        }
    }

    #[test]
    fn the_isomorphisms_number_the_automorphisms() {
        // Aut(E) is mu_k for k of 6, 4 and 2 as j is 0, 1728 and anything else, so
        // over F_13, where mu_6 is rational, the counts differ; over F_11, where only
        // mu_2 is, they do not.
        for (prime, expected) in [(13u32, [6usize, 4, 2]), (11, [2, 2, 2])] {
            let f = Integers::modulo(prime);
            let curves = [(0i64, 1i64), (1, 0), (1, 1)].map(|(a, b)| {
                Curve::new(f.clone(), f.element(a), f.element(b)).expect("nonsingular")
            });
            for (c, count) in curves.iter().zip(expected) {
                let automorphisms = c.isomorphisms(c);
                assert_eq!(
                    automorphisms.len(),
                    count,
                    "F_{prime}, Aut of {:?}",
                    c.coefficients()
                );
                for (i, iso) in automorphisms.iter().enumerate() {
                    assert!(
                        !automorphisms[i + 1..]
                            .iter()
                            .any(|other| other.u() == iso.u()),
                        "F_{prime}: two automorphisms share a u"
                    );
                    // Each sends the curve to itself, so every image is a point of it.
                    for p in all_points(&f, c) {
                        assert!(Arc::ptr_eq(iso.evaluate(&p).curve(), c));
                    }
                }
            }
        }
    }

    #[test]
    fn the_inverse_undoes_every_isomorphism() {
        for prime in [11u32, 13] {
            let f = Integers::modulo(prime);
            let curves = all_curves(&f);
            for (i, x) in curves.iter().enumerate() {
                for y in curves.iter().skip(i) {
                    for iso in x.isomorphisms(y) {
                        let back = iso.inverse();
                        assert!(Arc::ptr_eq(back.domain(), y), "F_{prime}");
                        assert!(Arc::ptr_eq(back.codomain(), x), "F_{prime}");
                        for p in all_points(&f, x) {
                            assert_eq!(back.evaluate(&iso.evaluate(&p)), p, "F_{prime}, {p}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_twist_becomes_isomorphic_over_the_quadratic_extension() {
        // 2 is not a square modulo 11, so the twist by it of y^2 = x^3 + x + 6 is
        // (2^2, 2^3 * 6) = (4, 4), which is not isomorphic to it over F_11.
        let f = Integers::modulo(11);
        assert_eq!(f.legendre_symbol(&f.element(2)), -1);
        let c = Curve::new(f.clone(), f.element(1), f.element(6)).expect("nonsingular");
        let twist = Curve::new(f.clone(), f.element(4), f.element(4)).expect("nonsingular");
        assert_eq!(
            c.j_invariant(),
            twist.j_invariant(),
            "a twist shares the j-invariant"
        );
        assert!(
            c.isomorphism(&twist).is_none(),
            "u would be a square root of 2"
        );

        // Upstairs the square root is there, and u^2 is the non-square that twisted.
        let big = crate::finite_fields::finite_field(11, 2);
        let over = |a: i64, b: i64| {
            Curve::new(big.clone(), big.from_integer(a), big.from_integer(b)).expect("nonsingular")
        };
        let iso = over(1, 6)
            .isomorphism(&over(4, 4))
            .expect("twists are isomorphic over the quadratic extension");
        assert_eq!(big.pow(iso.u(), 2), big.from_integer(2));
        // Frobenius moves u, so it is not in F_11.
        assert_ne!(big.pow(iso.u(), 11), iso.u().clone());
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
    fn the_classic_supersingular_families() {
        for prime in [5u32, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43] {
            let f = Integers::modulo(prime);
            let curve = |a: i64, b: i64| {
                Curve::new(f.clone(), f.element(a), f.element(b)).expect("nonsingular")
            };
            assert_eq!(
                curve(1, 0).is_supersingular(),
                prime % 4 == 3,
                "y^2 = x^3 + x over F_{prime}"
            );
            assert_eq!(
                curve(0, 1).is_supersingular(),
                prime % 3 == 2,
                "y^2 = x^3 + 1 over F_{prime}"
            );
        }
    }

    #[test]
    fn supersingularity_outlives_a_base_change_that_moves_the_trace() {
        // y^2 = x^3 + x over F_7 is supersingular with t = 0. Over F_49 the trace is
        // t^2 - 2p = -14, so the test has to be divisibility rather than vanishing.
        let f = crate::finite_fields::finite_field(7, 2);
        let c = Curve::new(f.clone(), f.identity(), f.zero()).expect("nonsingular");
        assert_eq!(c.order(), BigInt::from(64));
        assert_eq!(c.trace_of_frobenius(), BigInt::from(-14));
        assert!(c.is_supersingular());

        let base = Integers::modulo(7);
        let over_f7 =
            Curve::new(base.clone(), base.element(1), base.element(0)).expect("nonsingular");
        assert_eq!(over_f7.trace_of_frobenius(), BigInt::from(0));
        assert!(over_f7.is_supersingular());
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

    /// Every affine point, over any finite field.
    fn affine_points<F>(c: &Arc<Curve<F>>) -> Vec<Point<F>>
    where
        F: FiniteField + Clone,
        F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
    {
        let field = c.field();
        let q = field.order();
        let mut points = Vec::new();
        let mut i = BigInt::from(0u32);
        while i < q {
            let x = field.element_at(&i);
            let mut j = BigInt::from(0u32);
            while j < q {
                if let Some(p) = c.point(x.clone(), field.element_at(&j)) {
                    points.push(p);
                }
                j += 1u32;
            }
            i += 1u32;
        }
        points
    }

    /// `x([n]P) = phi_n(x) / psi_n^2(x)` against the group law itself, for every point
    /// and every `n` up to `up_to`. Cross-multiplied, so the field needs no division.
    fn check_the_x_map<F>(c: &Arc<Curve<F>>, up_to: usize)
    where
        F: FiniteField + Clone,
        F::E: RingOps + Eq + fmt::Debug + fmt::Display + for<'x> MulAssign<&'x F::E>,
    {
        let field = c.field();
        let points = affine_points(c);
        for n in 1..=up_to {
            let (phi, psi_squared) = c.multiplication_map(n);
            for p in &points {
                let (x, _) = p.coordinates().expect("affine");
                let denominator = psi_squared.evaluate(x);
                match c.multiply(n, p).coordinates() {
                    // [n]P is at infinity exactly where psi_n^2 vanishes.
                    None => assert!(field.is_zero(&denominator), "[{n}]{p} = O, psi^2 did not"),
                    Some((target, _)) => {
                        assert!(
                            !field.is_zero(&denominator),
                            "psi^2 vanished at {p}, [{n}]P did not"
                        );
                        assert_eq!(phi.evaluate(x), target.clone() * denominator, "x([{n}]{p})");
                    }
                }
            }
        }
    }

    #[test]
    fn the_x_map_agrees_with_the_group_law() {
        let f = Integers::modulo(11);
        // 13 points of prime order, so for n <= 7 nothing is n-torsion and psi_n^2
        // vanishes nowhere: this curve exercises only the affine branch.
        check_the_x_map(&curve().1, 7);
        // x^3 - x splits, so three points of order two and psi_n^2 does vanish.
        check_the_x_map(
            &Curve::new(f.clone(), f.element(-1), f.element(0)).expect("nonsingular"),
            6,
        );
        // Supersingular, 8 points over F_7.
        let f7 = Integers::modulo(7);
        check_the_x_map(
            &Curve::new(f7.clone(), f7.element(1), f7.element(0)).expect("nonsingular"),
            6,
        );
    }

    #[test]
    fn the_x_map_agrees_with_the_group_law_over_an_extension() {
        let f = crate::finite_fields::finite_field(5, 2);
        let c = Curve::new(
            f.clone(),
            f.element_at(&1u32.into()),
            f.element_at(&1u32.into()),
        )
        .expect("nonsingular");
        check_the_x_map(&c, 4);
    }

    #[test]
    fn division_polynomials_have_the_degrees_and_leading_coefficients_they_should() {
        let f = Integers::modulo(101);
        let c = Curve::new(f.clone(), f.element(3), f.element(5)).expect("nonsingular");
        let g = c.division_polynomials(9);

        assert_eq!(g.len(), 10);
        assert_eq!(g[0].degree(), None, "g_0 is zero");
        for (n, g_n) in g.iter().enumerate().skip(1) {
            // deg psi_n^2 = n^2 - 1, and the even ones give three of that to f.
            let expected = if n % 2 == 1 {
                (n * n - 1) / 2
            } else {
                (n * n - 4) / 2
            };
            assert_eq!(g_n.degree(), Some(expected), "deg g_{n}");
            assert_eq!(g_n.lead(), Some(&f.element(n as i64)), "lead g_{n}");
        }

        // One at a time gives the same list, and shares its ring.
        for (n, g_n) in g.iter().enumerate() {
            assert_eq!(&c.division_polynomial(n), g_n, "g_{n}");
        }
        // Shorter lists are prefixes.
        assert_eq!(c.division_polynomials(3), g[..4]);
    }

    #[test]
    fn phi_is_monic_of_degree_n_squared() {
        let f = Integers::modulo(101);
        let c = Curve::new(f.clone(), f.element(3), f.element(5)).expect("nonsingular");
        for n in 1..=7usize {
            let (phi, psi_squared) = c.multiplication_map(n);
            assert_eq!(phi.degree(), Some(n * n), "deg phi_{n}");
            assert_eq!(phi.lead(), Some(&f.identity()), "phi_{n} is monic");
            assert_eq!(psi_squared.degree(), Some(n * n - 1), "deg psi_{n}^2");
            assert_eq!(
                psi_squared.lead(),
                Some(&f.element((n * n) as i64)),
                "lead psi_{n}^2"
            );
        }
    }

    #[test]
    fn the_cubic_is_the_curve_equation() {
        let (f, c) = curve();
        let cubic = c.cubic();
        for x in 0..11i64 {
            let x = f.element(x);
            assert_eq!(cubic.evaluate(&x), c.evaluate(&x));
        }
        // Its roots are the points of order two, where y = 0.
        let two_torsion = Curve::new(f.clone(), f.element(-1), f.element(0)).expect("nonsingular");
        for x in [0i64, 1, 10] {
            assert!(f.is_zero(&two_torsion.cubic().evaluate(&f.element(x))));
        }
    }

    /// The formal derivative, which is what `r2` has to agree with.
    fn derivative<F>(ring: &Arc<PolynomialRing<F>>, p: &Polynomial<F>) -> Polynomial<F>
    where
        F: Field + Clone,
        F::E: RingOps + Eq,
    {
        let field = ring.coefficients();
        let coefficients: Vec<F::E> = p
            .coefficients()
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| field.from_integer(i) * c.clone())
            .collect();
        ring.element(coefficients)
    }

    /// A point of order exactly `n`, found by trying every one: `n` kills it and no
    /// `n / q` for a prime `q` dividing `n` does. The order is what `isogeny` trusts
    /// the caller for, so the tests work it out for themselves.
    fn point_of_order<F>(c: &Arc<Curve<F>>, n: usize) -> Option<Point<F>>
    where
        F: FiniteField + Clone,
        F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
    {
        let primes: Vec<usize> = crate::number_theory::factorise(n)
            .iter()
            .map(|(prime, _)| prime.try_into().expect("a factor of n fits where n does"))
            .collect();
        affine_points(c).into_iter().find(|p| {
            c.multiply(n, p).is_infinity()
                && primes
                    .iter()
                    .all(|prime| !c.multiply(n / prime, p).is_infinity())
        })
    }

    /// Everything an isogeny of degree `n` with kernel `<generator>` has to satisfy,
    /// over every point of the domain. Returns it, for any further checking.
    fn check_isogeny<F>(c: &Arc<Curve<F>>, generator: &Point<F>, n: usize) -> Isogeny<F>
    where
        F: FiniteField + Clone + fmt::Debug,
        F::E: RingOps + Eq + fmt::Debug + fmt::Display + for<'x> MulAssign<&'x F::E>,
        i64: Mul<F::E, Output = F::E>,
    {
        let field = c.field();
        let iso = c.isogeny(generator, n);
        let codomain = iso.codomain();
        assert_eq!(iso.degree(), n);
        assert!(
            !field.is_zero(&codomain.discriminant()),
            "the codomain is singular"
        );

        let mut points = affine_points(c);
        points.push(c.infinity());
        let kernel: Vec<Point<F>> = (0..n).map(|k| c.multiply(k, generator)).collect();
        let images: Vec<Point<F>> = points.iter().map(|p| iso.evaluate(p)).collect();

        for (i, (p, image)) in points.iter().zip(&images).enumerate() {
            assert_eq!(
                image.is_infinity(),
                kernel.contains(p),
                "{p} against the kernel"
            );
            if let Some((x, y)) = image.coordinates() {
                assert!(codomain.contains(x, y), "{p} maps off the codomain");
            }
            // Additivity is symmetric, so the unordered pairs are enough.
            for (q, other) in points.iter().zip(&images).skip(i) {
                assert_eq!(
                    iso.evaluate(&(p.clone() + q.clone())),
                    image.clone() + other.clone(),
                    "additivity at {p}, {q}"
                );
            }
            // The kernel being rational, every fibre is a coset of it.
            assert_eq!(
                images.iter().filter(|i| *i == image).count(),
                n,
                "the fibre over the image of {p}"
            );
        }

        // An isogeny over the base field preserves the number of rational points.
        assert_eq!(codomain.order(), c.order());
        assert_eq!(codomain.trace_of_frobenius(), c.trace_of_frobenius());

        let r1 = iso.x_map();
        let (numerator, denominator) = (r1.numerator(), r1.denominator());
        assert_eq!(numerator.degree(), Some(n), "deg of the numerator of r1");
        assert_eq!(
            denominator.degree(),
            Some(n - 1),
            "deg of the kernel polynomial"
        );

        let ring = c.polynomials();
        let (d_numerator, d_denominator) =
            (derivative(ring, numerator), derivative(ring, denominator));
        for (p, image) in points.iter().zip(&images) {
            let Some((x, y)) = p.coordinates() else {
                continue;
            };
            let d = denominator.evaluate(x);
            match image.coordinates() {
                None => assert!(field.is_zero(&d), "{p} is in the kernel, r1 has no pole"),
                Some((image_x, image_y)) => {
                    assert_eq!(
                        numerator.evaluate(x),
                        image_x.clone() * d.clone(),
                        "r1 at {p}"
                    );
                    // r2 = r1', cross-multiplied by the square of the denominator.
                    assert_eq!(
                        image_y.clone() * d.clone() * d.clone(),
                        y.clone()
                            * (d_numerator.evaluate(x) * d.clone()
                                - numerator.evaluate(x) * d_denominator.evaluate(x)),
                        "r2 at {p}"
                    );
                }
            }
        }
        iso
    }

    #[test]
    fn velu_in_degrees_two_three_and_five() {
        let f = Integers::modulo(11);
        // The search lands on y^2 = x^3 + 1 for degrees two and three, and on
        // y^2 = x^3 + x + 7 for five. Degree two is a kernel of 2-torsion, where v_Q
        // takes its other branch.
        for n in [2usize, 3, 5] {
            let (c, p) = all_curves(&f)
                .into_iter()
                .find_map(|c| point_of_order(&c, n).map(|p| (c, p)))
                .unwrap_or_else(|| panic!("no curve over F_11 has a point of order {n}"));
            check_isogeny(&c, &p, n);
        }
    }

    #[test]
    fn a_two_isogeny_worked_out_by_hand() {
        // y^2 = x^3 - x over F_11 with kernel {O, (1, 0)}: g_T = 3 + a = 2, taken
        // through the 2-torsion branch, so v = 2 and u = 0. Then w = x_T * v = 2,
        // leaving a - 5v = 0 and b - 7w = -14 = 8.
        let f = Integers::modulo(11);
        let c = Curve::new(f.clone(), f.element(-1), f.element(0)).expect("nonsingular");
        let t = c.point(f.element(1), f.element(0)).expect("on the curve");
        let iso = c.isogeny(&t, 2);
        assert_eq!(
            iso.codomain().coefficients(),
            (&f.element(0), &f.element(8))
        );

        // r1(x) = x + 2/(x - 1) and r2(x) = 1 - 2/(x - 1)^2, so (4, 4) goes to
        // (4 + 2 * 4, 4 * 2).
        let image = iso.evaluate(&c.point(f.element(4), f.element(4)).expect("on the curve"));
        assert_eq!(image.coordinates(), Some((&f.element(1), &f.element(8))));
    }

    #[test]
    fn a_three_isogeny_worked_out_by_hand() {
        // y^2 = x^3 + 1 over F_11 with kernel {O, (0, 1), (0, 10)}, whose two affine
        // points are a pair, so S is just (0, 1). There g_Q = 3*0 + 0 = 0, leaving
        // v = 0 and u = 4, so a stays 0 and b becomes 1 - 7*4 = 6.
        let f = Integers::modulo(11);
        let c = Curve::new(f.clone(), f.element(0), f.element(1)).expect("nonsingular");
        let g = c.point(f.element(0), f.element(1)).expect("on the curve");
        let iso = c.isogeny(&g, 3);
        assert_eq!(
            iso.codomain().coefficients(),
            (&f.element(0), &f.element(6))
        );

        // r1(x) = x + 4/x^2 and r2(x) = 1 - 8/x^3, so (5, 4) goes to (5 + 4*4, 4*10).
        let image = iso.evaluate(&c.point(f.element(5), f.element(4)).expect("on the curve"));
        assert_eq!(image.coordinates(), Some((&f.element(10), &f.element(7))));
    }

    #[test]
    fn an_isogeny_of_a_supersingular_curve_is_supersingular() {
        // y^2 = x^3 + x over F_49, which is supersingular with 64 points, so every
        // order is a power of two. Degree four holds both a 2-torsion point and a
        // pair, which exercises both branches of v_Q in one isogeny.
        let f = crate::finite_fields::finite_field(7, 2);
        let c = Curve::new(
            f.clone(),
            f.element_at(&1u32.into()),
            f.element_at(&0u32.into()),
        )
        .expect("nonsingular");
        assert!(c.is_supersingular());

        for n in [2usize, 4] {
            let p = point_of_order(&c, n).unwrap_or_else(|| panic!("a point of order {n}"));
            let iso = check_isogeny(&c, &p, n);
            assert!(iso.codomain().is_supersingular(), "degree {n}");
        }
    }

    #[test]
    #[should_panic(expected = "the generator's order is larger than 7")]
    fn an_order_too_small_is_rejected() {
        let (f, c) = curve();
        // The group has prime order 13, so 7 * P is not infinity.
        let g = c.point(f.element(2), f.element(7)).expect("on the curve");
        c.isogeny(&g, 7);
    }

    #[test]
    #[should_panic(expected = "the generator's order is larger than 0")]
    fn order_zero_is_rejected() {
        // Infinity has order one, which leaves nothing for zero to be the order of.
        let (_, c) = curve();
        c.isogeny(&c.infinity(), 0);
    }

    #[test]
    #[should_panic(expected = "the generator's order is smaller than the order given")]
    fn an_order_too_large_is_rejected() {
        // (1, 0) has order two on y^2 = x^3 - x, so the walk over a kernel of four
        // reaches infinity halfway.
        let f = Integers::modulo(11);
        let c = Curve::new(f.clone(), f.element(-1), f.element(0)).expect("nonsingular");
        let t = c.point(f.element(1), f.element(0)).expect("on the curve");
        c.isogeny(&t, 4);
    }

    #[test]
    fn the_trivial_kernel_gives_the_identity_map() {
        let (_, c) = curve();
        let iso = c.isogeny(&c.infinity(), 1);

        assert_eq!(iso.degree(), 1);
        assert!(iso.summands().is_empty());
        assert_eq!(iso.codomain().coefficients(), c.coefficients());
        assert!(iso.evaluate(&c.infinity()).is_infinity());
        for p in affine_points(&c) {
            assert_eq!(iso.evaluate(&p).coordinates(), p.coordinates(), "{p}");
        }

        // r1 = x, of degree one over degree zero.
        let r1 = iso.x_map();
        assert_eq!(r1.numerator(), &c.polynomials().indeterminate());
        assert_eq!(r1.denominator().degree(), Some(0));
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
