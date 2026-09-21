//! Algebraic structure traits and the [`QuotientRing`] construction.
//!
//! The hierarchy goes `Domain` → `Semigroup` → `Monoid` → `SemiRing` → `Ring` → `EuclideanDomain`,
//! and (orthogonally) `CommutativeMonoid` → `AdditiveGroup`. Traits are implemented on a
//! value-domain handle (e.g. [`crate::integers::Integers`]) rather than on the element type
//! itself, so a single element type can participate in multiple structures.

use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

/// The ring operations, bundled so the element bound can be written once per `where`
/// clause instead of spelled out three times.
///
/// These are supertraits rather than a `where` clause, which is what makes them usable
/// at the call site: Rust elaborates supertrait bounds, but never a trait's own `where`
/// clause, so `R::E: RingOps` yields `Add`/`Sub`/`Mul` while `R: Ring` alone does not.
/// The blanket impl covers every element type that already has the three operators.
pub trait RingOps: Sized + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {}

impl<T: Add<Output = T> + Sub<Output = T> + Mul<Output = T>> RingOps for T {}

/// A value-domain handle: it names a concrete element type and knows how to build
/// elements of it from representatives.
pub trait Domain {
    /// The element type carried by this domain.
    ///
    /// Note the absence of `Eq`: it is required further up, at [`EuclideanDomain`] and
    /// [`Field`], where algorithms actually test for zero. Element types without a
    /// decidable equality (constructive reals, say) can still form the lower structures.
    type E: Clone;

    /// A representative that this domain knows how to turn into an element. Element types
    /// that carry a ring handle can't implement `From`, so the domain supplies the conversion.
    type Repr;

    /// Builds an element of this domain from a representative.
    fn element<T: Into<Self::Repr>>(&self, value: T) -> Self::E;

    /// Tests whether two representatives denote the same element of this domain.
    /// Available only where the element type has a decidable equality.
    fn eq<F: Into<Self::Repr>, G: Into<Self::Repr>>(&self, a: F, b: G) -> bool
    where
        Self::E: Eq,
    {
        self.element(a) == self.element(b)
    }
}

/// A domain whose elements form a semigroup under multiplication.
pub trait Semigroup: Domain
where
    Self::E: Mul<Output = Self::E>,
{
}

/// A semigroup with a multiplicative identity.
pub trait Monoid: Semigroup
where
    Self::E: Mul<Output = Self::E>,
{
    /// Returns the multiplicative identity (`1`).
    fn identity(&self) -> Self::E;
}

/// A domain whose elements form a commutative monoid under addition.
pub trait CommutativeMonoid: Domain
where
    Self::E: Add<Output = Self::E>,
{
    /// Returns the additive identity (`0`).
    fn zero(&self) -> Self::E;
}

/// A commutative monoid under addition where every element has an additive inverse,
/// so subtraction is total.
pub trait AdditiveGroup: CommutativeMonoid
where
    Self::E: Add<Output = Self::E> + Sub<Output = Self::E>,
{
}

/// A domain that is both a multiplicative monoid and a commutative additive monoid.
pub trait SemiRing: Monoid + CommutativeMonoid
where
    Self::E: Add<Output = Self::E> + Mul<Output = Self::E>,
{
}

/// A ring: a [`SemiRing`] whose additive monoid is also a group.
pub trait Ring: SemiRing + AdditiveGroup
where
    Self::E: RingOps,
{
}

/// Elements that support Euclidean division, yielding `(quotient, remainder)`.
pub trait DivRem: Sized {
    /// Returns `(self / divisor, self % divisor)`.
    fn div_rem(&self, divisor: &Self) -> (Self, Self);
}

/// A Euclidean domain: a ring in which every element supports [`DivRem`] and the
/// domain has a known unit structure (needed to canonicalise gcds).
///
/// `x = unit_part(x) * normal_part(x)`, where `normal_part(x) = unit_inverse(unit_part(x)) * x`
/// is the canonical representative of the associate class of `x`. For `Z`, the unit
/// part is the sign and the normal part is the absolute value. For a polynomial
/// ring `F[x]` over a field, the unit part is the leading coefficient and the
/// normal part is the monic associate.
pub trait EuclideanDomain: Ring
where
    Self::E: RingOps + DivRem + Eq,
{
    /// Returns the unit part of `x` (the unit that, when divided out, leaves the
    /// canonical representative of `x`'s associate class).
    ///
    /// Convention: `unit_part(zero) = identity` so that `zero / unit_part(zero) = zero`.
    fn unit_part(&self, x: &Self::E) -> Self::E;

    /// Returns the multiplicative inverse of a unit. The result is only meaningful
    /// when the input is a unit of the domain (e.g. produced by [`Self::unit_part`]).
    fn unit_inverse(&self, u: &Self::E) -> Self::E;

    /// Iterative extended Euclidean algorithm. Returns `(gcd, x, y)` such that
    /// `x * a + y * b = gcd`, with `gcd` canonicalised via [`Self::unit_part`] so
    /// that coprime inputs always yield `gcd == identity`.
    fn extended_gcd(&self, a: Self::E, b: Self::E) -> (Self::E, Self::E, Self::E) {
        let (mut old_r, mut r) = (a, b);
        let (mut old_s, mut s) = (self.identity(), self.zero());
        let (mut old_t, mut t) = (self.zero(), self.identity());
        let zero = self.zero();
        while r != zero {
            let (q, new_r) = old_r.div_rem(&r);
            old_r = std::mem::replace(&mut r, new_r);
            let new_s = old_s - q.clone() * s.clone();
            old_s = std::mem::replace(&mut s, new_s);
            let new_t = old_t - q * t.clone();
            old_t = std::mem::replace(&mut t, new_t);
        }
        let u_inv = self.unit_inverse(&self.unit_part(&old_r));
        (
            old_r * u_inv.clone(),
            old_s * u_inv.clone(),
            old_t * u_inv,
        )
    }
}

/// A field: a ring in which every nonzero element has a multiplicative inverse.
pub trait Field: Ring
where
    Self::E: RingOps + Eq,
{
    /// Returns the multiplicative inverse of `x`, or [`None`] if `x` is the additive identity.
    fn inverse(&self, x: &Self::E) -> Option<Self::E>;
}

/// An element of a quotient ring `R / (m)`.
///
/// The stored representative is reduced modulo `m` after construction and after every
/// arithmetic operation. Two elements compare equal iff their representatives are
/// congruent modulo `m` (so non-canonical sign conventions in [`DivRem`] don't break
/// equality).
#[derive(Debug)]
pub struct QuotientRingElement<R: EuclideanDomain>
where
    R::E: RingOps + DivRem + Eq,
{
    value: R::E,
    ring: Rc<QuotientRing<R>>,
}

/// The quotient ring `R / (modulus)`, where `R` is a Euclidean domain.
#[derive(Debug, Clone)]
pub struct QuotientRing<R: EuclideanDomain>
where
    R::E: RingOps + DivRem + Eq,
{
    ring: R,
    modulus: R::E,
}

impl<R> QuotientRing<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    /// Construct `R / (modulus)`, wrapped in an [`Rc`] so that elements can hold a
    /// cheap shared reference back to their containing ring.
    pub fn new(ring: R, modulus: R::E) -> Rc<Self> {
        Rc::new(QuotientRing { ring, modulus })
    }

    /// The modulus generating the ideal `(modulus)` that this quotient is taken by.
    pub fn modulus(&self) -> &R::E {
        &self.modulus
    }

    fn reduce(&self, value: R::E) -> R::E {
        value.div_rem(&self.modulus).1
    }

    fn is_zero(&self, a: &R::E) -> bool {
        a.div_rem(&self.modulus).1 == self.ring.zero()
    }
}

impl<R> Clone for QuotientRingElement<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn clone(&self) -> Self {
        QuotientRingElement {
            value: self.value.clone(),
            ring: Rc::clone(&self.ring),
        }
    }
}

impl<R> PartialEq for QuotientRingElement<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        let diff = self.value.clone() - other.value.clone();
        self.ring.is_zero(&diff)
    }
}

impl<R> Eq for QuotientRingElement<R>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

/// Arithmetic over every owned/borrowed operand combination.
///
/// The borrowed forms need the underlying ring's elements to support reference ops,
/// so `&a + &b` builds the result without copying either operand.
macro_rules! quotient_ring_ops {
    ($($op:ident, $method:ident);* $(;)?) => {$(
        impl<R> $op for QuotientRingElement<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
        {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                debug_assert!(
                    Rc::ptr_eq(&self.ring, &rhs.ring),
                    "QuotientRingElement operands belong to different quotient rings"
                );
                let value = self.ring.reduce(self.value.$method(rhs.value));
                QuotientRingElement { value, ring: self.ring }
            }
        }

        impl<R> $op<&QuotientRingElement<R>> for &QuotientRingElement<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
            for<'c> &'c R::E: $op<&'c R::E, Output = R::E>,
        {
            type Output = QuotientRingElement<R>;
            fn $method(self, rhs: &QuotientRingElement<R>) -> Self::Output {
                debug_assert!(
                    Rc::ptr_eq(&self.ring, &rhs.ring),
                    "QuotientRingElement operands belong to different quotient rings"
                );
                let value = self.ring.reduce((&self.value).$method(&rhs.value));
                QuotientRingElement { value, ring: Rc::clone(&self.ring) }
            }
        }

        impl<R> $op<&QuotientRingElement<R>> for QuotientRingElement<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
            for<'c> &'c R::E: $op<&'c R::E, Output = R::E>,
        {
            type Output = QuotientRingElement<R>;
            fn $method(self, rhs: &QuotientRingElement<R>) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl<R> $op<QuotientRingElement<R>> for &QuotientRingElement<R>
        where
            R: EuclideanDomain,
            R::E: RingOps + DivRem + Eq,
            for<'c> &'c R::E: $op<&'c R::E, Output = R::E>,
        {
            type Output = QuotientRingElement<R>;
            fn $method(self, rhs: QuotientRingElement<R>) -> Self::Output {
                self.$method(&rhs)
            }
        }
    )*};
}
quotient_ring_ops!(Add, add; Sub, sub; Mul, mul);

impl<R> Domain for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    type E = QuotientRingElement<R>;
    type Repr = R::E;

    /// Reduces the representative modulo [`QuotientRing::modulus`].
    fn element<T: Into<R::E>>(&self, value: T) -> Self::E {
        QuotientRingElement {
            value: self.reduce(value.into()),
            ring: Rc::clone(self),
        }
    }
}

impl<R> Semigroup for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

impl<R> Monoid for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn identity(&self) -> Self::E {
        self.element(self.ring.identity())
    }
}

impl<R> CommutativeMonoid for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn zero(&self) -> Self::E {
        self.element(self.ring.zero())
    }
}

impl<R> AdditiveGroup for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

impl<R> SemiRing for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

impl<R> Ring for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
}

/// `Field` impl on a generic quotient ring.
///
/// The algebraic contract (every nonzero element has an inverse) only holds when
/// the modulus is irreducible in `R`. With a reducible modulus, [`Field::inverse`]
/// safely returns `None` for non-units instead of producing a wrong inverse.
///
/// Relies on [`EuclideanDomain::extended_gcd`] returning a canonical gcd, so
/// `gcd == identity` is the right coprimality test.
impl<R> Field for Rc<QuotientRing<R>>
where
    R: EuclideanDomain,
    R::E: RingOps + DivRem + Eq,
{
    fn inverse(&self, x: &Self::E) -> Option<Self::E> {
        if x == &self.zero() {
            return None;
        }
        let (gcd, s, _) = self.ring.extended_gcd(x.value.clone(), self.modulus.clone());
        if gcd != self.ring.identity() {
            return None;
        }
        Some(self.element(s))
    }
}
