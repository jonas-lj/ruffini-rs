//! Prime fields whose modulus is fixed at compile time, held in four 64-bit limbs.
//!
//! [`crate::structures::QuotientRing`] over [`crate::integers::Integers`] is the
//! general `Z/n`, and it stays on arbitrary precision: every element is a heap
//! allocation and every operation a general division. Here the modulus is a constant
//! instead, which buys a fixed-size representation and Montgomery's reduction, where
//! the division becomes a multiplication by `-n^{-1} mod 2^64` and a shift.
//!
//! Elements are kept in Montgomery form, as `x * 2^256 mod n`, where a product
//! reduces with no division at all. The form is an implementation detail:
//! [`crate::structures::Domain::element`] takes an ordinary integer and [`Display`]
//! prints one.
//!
//! A new field is a modulus and nothing else, the two constants Montgomery's method
//! needs being derived from it:
//!
//! ```
//! use ruffini::fixed_fields::{Fp, Prime};
//! use ruffini::structures::{Domain, Monoid};
//!
//! struct Mersenne61;
//! impl Prime for Mersenne61 {
//!     const MODULUS: [u64; 4] = [(1 << 61) - 1, 0, 0, 0];
//! }
//!
//! let f = Fp::<Mersenne61>::new();
//! let x = f.element(1u64 << 60);
//! // 2^61 is one here, so squaring 2^60 wraps back down to 2^59.
//! assert_eq!(f.pow(&x, 2u32), f.element(1u64 << 59));
//! ```

use crate::finite_fields::FiniteField;
use crate::structures::{
    AdditiveGroup, CommutativeMonoid, DivRem, Domain, EuclideanDomain, Field, Monoid, Ring,
    SemiRing, Semigroup,
};
use num_bigint::BigInt;
use std::fmt;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub};

/// How many limbs a modulus has, and so how wide an element is.
const LIMBS: usize = 4;

/// The modulus of a [`Fp`], as 64-bit limbs with the least significant first.
///
/// Only [`Prime::MODULUS`] has to be given, and it has to be odd, which Montgomery's
/// method needs and a prime above two is. The other two constants come from it.
pub trait Prime {
    /// The modulus `n`, which has to be odd and above one.
    const MODULUS: [u64; LIMBS];

    /// `-n^{-1} mod 2^64`, the multiplier that clears a limb at a time.
    const INVERSE: u64 = inverse_of_the_least_limb(Self::MODULUS[0]);

    /// `2^512 mod n`, which multiplying by carries an integer into Montgomery form.
    const RADIX_SQUARED: [u64; LIMBS] = radix_squared(Self::MODULUS);
}

/// `-n^{-1} mod 2^64`, by Newton's iteration on `x -> x(2 - nx)`, which doubles the
/// number of correct bits and so needs six rounds from the one bit that `x = 1` has.
pub const fn inverse_of_the_least_limb(modulus: u64) -> u64 {
    assert!(modulus % 2 == 1, "the modulus has to be odd");
    let mut inverse: u64 = 1;
    let mut rounds = 0;
    while rounds < 6 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(modulus.wrapping_mul(inverse)));
        rounds += 1;
    }
    inverse.wrapping_neg()
}

/// `2^512 mod n`, by doubling one 512 times, which needs nothing wider than the
/// modulus itself.
pub const fn radix_squared(modulus: [u64; LIMBS]) -> [u64; LIMBS] {
    let mut value = [1, 0, 0, 0];
    let mut doublings = 0;
    while doublings < 128 * LIMBS {
        value = doubled(value, modulus);
        doublings += 1;
    }
    value
}

/// `2v mod n` for `v` below `n`, which one subtraction brings back under it.
const fn doubled(value: [u64; LIMBS], modulus: [u64; LIMBS]) -> [u64; LIMBS] {
    let mut shifted = [0u64; LIMBS];
    let mut carry = 0;
    let mut i = 0;
    while i < LIMBS {
        shifted[i] = (value[i] << 1) | carry;
        carry = value[i] >> 63;
        i += 1;
    }
    // Twice something below n is below 2n, so the result is in range once n is taken
    // off, and the top bit that fell into the carry goes with it.
    if carry != 0 || !is_below(shifted, modulus) {
        shifted = wrapping_difference(shifted, modulus);
    }
    shifted
}

/// Whether `a < b`, comparing from the most significant limb down.
const fn is_below(a: [u64; LIMBS], b: [u64; LIMBS]) -> bool {
    let mut i = LIMBS;
    while i > 0 {
        i -= 1;
        if a[i] != b[i] {
            return a[i] < b[i];
        }
    }
    false
}

/// `a - b` modulo `2^256`, which is `a - b` whenever that is not negative.
const fn wrapping_difference(a: [u64; LIMBS], b: [u64; LIMBS]) -> [u64; LIMBS] {
    let mut difference = [0u64; LIMBS];
    let mut borrow = 0u64;
    let mut i = 0;
    while i < LIMBS {
        let (first, one) = a[i].overflowing_sub(b[i]);
        let (second, two) = first.overflowing_sub(borrow);
        difference[i] = second;
        borrow = (one as u64) | (two as u64);
        i += 1;
    }
    difference
}

/// The field of integers modulo `P::MODULUS`.
///
/// A handle with nothing in it, so elements carry no reference back to it and copying
/// one is copying four words.
pub struct Fp<P>(PhantomData<P>);

/// An element of [`Fp`], as `x * 2^256 mod n` in four limbs.
pub struct FpElement<P> {
    limbs: [u64; LIMBS],
    prime: PhantomData<P>,
}

impl<P: Prime> Fp<P> {
    pub const fn new() -> Self {
        Fp(PhantomData)
    }

    /// The modulus, as an integer.
    pub fn modulus(&self) -> BigInt {
        from_limbs(&P::MODULUS)
    }
}

impl<P: Prime> Default for Fp<P> {
    fn default() -> Self {
        Fp::new()
    }
}

impl<P> Clone for Fp<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Copy for Fp<P> {}

impl<P> fmt::Debug for Fp<P>
where
    P: Prime,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Z/{}", from_limbs(&P::MODULUS))
    }
}

impl<P: Prime> FpElement<P> {
    /// The integer this stands for, which is the limbs taken out of Montgomery form.
    pub fn to_integer(&self) -> BigInt {
        from_limbs(&multiply::<P>(&self.limbs, &[1, 0, 0, 0]))
    }

    /// The limbs as stored, which is `x * 2^256 mod n` rather than `x`.
    pub fn montgomery_limbs(&self) -> &[u64; LIMBS] {
        &self.limbs
    }

    fn from_limbs(limbs: [u64; LIMBS]) -> Self {
        FpElement {
            limbs,
            prime: PhantomData,
        }
    }
}

/// `a + b mod n`, for `a` and `b` below `n`.
fn add<P: Prime>(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut sum = [0u64; LIMBS];
    let mut carry = 0u64;
    for i in 0..LIMBS {
        let total = a[i] as u128 + b[i] as u128 + carry as u128;
        sum[i] = total as u64;
        carry = (total >> 64) as u64;
    }
    // Below 2n, so one subtraction is enough, and the carry is the bit that did not
    // fit rather than a reason to stop.
    if carry != 0 || !is_below(sum, P::MODULUS) {
        sum = wrapping_difference(sum, P::MODULUS);
    }
    sum
}

/// `a - b mod n`, for `a` and `b` below `n`.
fn subtract<P: Prime>(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let difference = wrapping_difference(*a, *b);
    if is_below(*a, *b) {
        // It went negative, which modulo 2^256 is n short of the answer.
        add_wrapping(difference, P::MODULUS)
    } else {
        difference
    }
}

/// `a + b` modulo `2^256`, which is where a subtraction that went negative lands.
fn add_wrapping(a: [u64; LIMBS], b: [u64; LIMBS]) -> [u64; LIMBS] {
    let mut sum = [0u64; LIMBS];
    let mut carry = 0u64;
    for i in 0..LIMBS {
        let total = a[i] as u128 + b[i] as u128 + carry as u128;
        sum[i] = total as u64;
        carry = (total >> 64) as u64;
    }
    sum
}

/// `a * b * 2^-256 mod n`, which is the product of two Montgomery forms in Montgomery
/// form.
///
/// The product is taken in full and then reduced, a limb at a time: adding the
/// multiple of `n` that clears the lowest limb changes nothing modulo `n` and lets the
/// limb be shifted away, which is the division by `2^256` that the form asks for.
fn multiply<P: Prime>(a: &[u64; LIMBS], b: &[u64; LIMBS]) -> [u64; LIMBS] {
    let mut t = [0u64; 2 * LIMBS + 1];
    for i in 0..LIMBS {
        let mut carry = 0u64;
        for j in 0..LIMBS {
            let total = t[i + j] as u128 + a[j] as u128 * b[i] as u128 + carry as u128;
            t[i + j] = total as u64;
            carry = (total >> 64) as u64;
        }
        t[i + LIMBS] = carry;
    }

    for i in 0..LIMBS {
        let m = t[i].wrapping_mul(P::INVERSE);
        let mut carry = 0u64;
        for j in 0..LIMBS {
            let total = t[i + j] as u128 + m as u128 * P::MODULUS[j] as u128 + carry as u128;
            t[i + j] = total as u64;
            carry = (total >> 64) as u64;
        }
        // The carry runs on past the limbs the multiple reached.
        let mut k = i + LIMBS;
        while carry != 0 {
            let total = t[k] as u128 + carry as u128;
            t[k] = total as u64;
            carry = (total >> 64) as u64;
            k += 1;
        }
    }

    let mut result = [0u64; LIMBS];
    result.copy_from_slice(&t[LIMBS..2 * LIMBS]);
    // What is left is below 2n, the top bit of it in the last limb.
    if t[2 * LIMBS] != 0 || !is_below(result, P::MODULUS) {
        result = wrapping_difference(result, P::MODULUS);
    }
    result
}

/// The limbs as an integer, least significant limb first.
fn from_limbs(limbs: &[u64; LIMBS]) -> BigInt {
    let mut value = BigInt::from(0u32);
    for limb in limbs.iter().rev() {
        value <<= 64;
        value += BigInt::from(*limb);
    }
    value
}

/// An integer as limbs, reduced modulo `n` first, so any integer is accepted.
fn to_limbs<P: Prime>(value: &BigInt) -> [u64; LIMBS] {
    let modulus = from_limbs(&P::MODULUS);
    let mut reduced = value % &modulus;
    if reduced < BigInt::from(0u32) {
        reduced += &modulus;
    }
    let mut limbs = [0u64; LIMBS];
    let mask = BigInt::from(u64::MAX);
    for limb in limbs.iter_mut() {
        let digit = &reduced & &mask;
        *limb = u64::try_from(digit).expect("a 64-bit mask leaves 64 bits");
        reduced >>= 64;
    }
    limbs
}

impl<P> Clone for FpElement<P> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<P> Copy for FpElement<P> {}

/// The Montgomery form of an element is unique once reduced, so the limbs decide it.
impl<P> PartialEq for FpElement<P> {
    fn eq(&self, other: &Self) -> bool {
        self.limbs == other.limbs
    }
}

impl<P> Eq for FpElement<P> {}

impl<P: Prime> fmt::Display for FpElement<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.to_integer().fmt(f)
    }
}

impl<P: Prime> fmt::Debug for FpElement<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_integer())
    }
}

/// The operators, owned and borrowed, which is what the generic algorithms ask of an
/// element type.
macro_rules! fp_binop {
    ($op:ident, $method:ident, $apply:ident) => {
        impl<P: Prime> $op for FpElement<P> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                FpElement::from_limbs($apply::<P>(&self.limbs, &rhs.limbs))
            }
        }

        impl<P: Prime> $op<&FpElement<P>> for FpElement<P> {
            type Output = FpElement<P>;
            fn $method(self, rhs: &FpElement<P>) -> FpElement<P> {
                FpElement::from_limbs($apply::<P>(&self.limbs, &rhs.limbs))
            }
        }

        impl<P: Prime> $op<FpElement<P>> for &FpElement<P> {
            type Output = FpElement<P>;
            fn $method(self, rhs: FpElement<P>) -> FpElement<P> {
                FpElement::from_limbs($apply::<P>(&self.limbs, &rhs.limbs))
            }
        }

        impl<P: Prime> $op<&FpElement<P>> for &FpElement<P> {
            type Output = FpElement<P>;
            fn $method(self, rhs: &FpElement<P>) -> FpElement<P> {
                FpElement::from_limbs($apply::<P>(&self.limbs, &rhs.limbs))
            }
        }
    };
}
fp_binop!(Add, add, add);
fp_binop!(Sub, sub, subtract);
fp_binop!(Mul, mul, multiply);

impl<P: Prime> AddAssign for FpElement<P> {
    fn add_assign(&mut self, rhs: Self) {
        self.limbs = add::<P>(&self.limbs, &rhs.limbs);
    }
}

impl<P: Prime> AddAssign<&FpElement<P>> for FpElement<P> {
    fn add_assign(&mut self, rhs: &FpElement<P>) {
        self.limbs = add::<P>(&self.limbs, &rhs.limbs);
    }
}

impl<P: Prime> MulAssign for FpElement<P> {
    fn mul_assign(&mut self, rhs: Self) {
        self.limbs = multiply::<P>(&self.limbs, &rhs.limbs);
    }
}

impl<P: Prime> MulAssign<&FpElement<P>> for FpElement<P> {
    fn mul_assign(&mut self, rhs: &FpElement<P>) {
        self.limbs = multiply::<P>(&self.limbs, &rhs.limbs);
    }
}

impl<P: Prime> Neg for FpElement<P> {
    type Output = Self;
    fn neg(self) -> Self {
        FpElement::from_limbs(subtract::<P>(&[0; LIMBS], &self.limbs))
    }
}

/// Division, which is multiplication by the inverse.
///
/// # Panics
/// If `rhs` is zero.
impl<P: Prime> Div for FpElement<P> {
    type Output = Self;
    // Dividing is multiplying by the inverse, which is what clippy is warning about.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Self) -> Self {
        let field = Fp::<P>::new();
        self * field.invert(&rhs).expect("division by zero")
    }
}

/// Exact division, the remainder always zero, which is what makes a field a Euclidean
/// domain.
impl<P: Prime> DivRem for FpElement<P> {
    /// # Panics
    /// If `divisor` is zero.
    fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let field = Fp::<P>::new();
        (*self / *divisor, field.zero())
    }
}

/// An integer on either side of an operator, embedded with [`Ring::from_integer`].
macro_rules! fp_integer_ops {
    ($($int:ty),+ $(,)?) => {$(
        impl<P: Prime> Mul<FpElement<P>> for $int {
            type Output = FpElement<P>;
            fn mul(self, rhs: FpElement<P>) -> FpElement<P> {
                Fp::<P>::new().from_integer(self) * rhs
            }
        }

        impl<P: Prime> Add<FpElement<P>> for $int {
            type Output = FpElement<P>;
            fn add(self, rhs: FpElement<P>) -> FpElement<P> {
                Fp::<P>::new().from_integer(self) + rhs
            }
        }

        impl<P: Prime> Mul<$int> for FpElement<P> {
            type Output = FpElement<P>;
            fn mul(self, rhs: $int) -> FpElement<P> {
                self * Fp::<P>::new().from_integer(rhs)
            }
        }

        impl<P: Prime> Add<$int> for FpElement<P> {
            type Output = FpElement<P>;
            fn add(self, rhs: $int) -> FpElement<P> {
                self + Fp::<P>::new().from_integer(rhs)
            }
        }

        impl<P: Prime> Sub<$int> for FpElement<P> {
            type Output = FpElement<P>;
            fn sub(self, rhs: $int) -> FpElement<P> {
                self - Fp::<P>::new().from_integer(rhs)
            }
        }
    )+};
}
fp_integer_ops!(i32, i64, u32, u64);

impl<P: Prime> FpElement<P> {
    /// `self^exp`, in `O(log exp)` multiplications. `exp == 0` gives one.
    ///
    /// # Panics
    /// If `exp` is negative.
    pub fn pow<T: Into<BigInt>>(&self, exp: T) -> Self {
        crate::pow::pow(&Fp::<P>::new(), self, exp)
    }
}

impl<P: Prime> Domain for Fp<P> {
    type E = FpElement<P>;
    type Repr = BigInt;

    /// Any integer, reduced modulo `n` and carried into Montgomery form.
    fn element<T: Into<Self::Repr>>(&self, value: T) -> Self::E {
        let limbs = to_limbs::<P>(&value.into());
        FpElement::from_limbs(multiply::<P>(&limbs, &P::RADIX_SQUARED))
    }
}

impl<P: Prime> Semigroup for Fp<P> {}

impl<P: Prime> Monoid for Fp<P> {
    /// One in Montgomery form, which is `2^256 mod n`.
    fn identity(&self) -> Self::E {
        self.element(BigInt::from(1u32))
    }
}

impl<P: Prime> CommutativeMonoid for Fp<P> {
    fn zero(&self) -> Self::E {
        FpElement::from_limbs([0; LIMBS])
    }

    /// Zero is zero in either form, so this needs no conversion.
    fn is_zero(&self, x: &Self::E) -> bool {
        x.limbs == [0; LIMBS]
    }
}

impl<P: Prime> AdditiveGroup for Fp<P> {}

impl<P: Prime> SemiRing for Fp<P> {}

impl<P: Prime> Ring for Fp<P> {}

impl<P: Prime> EuclideanDomain for Fp<P> {
    /// Every nonzero element is a unit of a field, so each is its own unit part.
    fn unit_part(&self, x: &Self::E) -> Self::E {
        if self.is_zero(x) {
            self.identity()
        } else {
            *x
        }
    }

    fn unit_inverse(&self, u: &Self::E) -> Self::E {
        self.invert(u).expect("a unit is not zero")
    }
}

impl<P: Prime> Field for Fp<P> {
    /// By Fermat, `x^(n-2)` is the inverse, which is a few hundred multiplications and
    /// no extended Euclid.
    fn invert(&self, x: &Self::E) -> Option<Self::E> {
        if self.is_zero(x) {
            return None;
        }
        Some(self.pow(x, self.modulus() - 2u32))
    }
}

impl<P: Prime> FiniteField for Fp<P> {
    fn order(&self) -> BigInt {
        self.modulus()
    }

    /// The modulus, the field being of prime order.
    fn characteristic(&self) -> BigInt {
        self.modulus()
    }

    /// The integer `index` itself, which for a prime field is the enumeration.
    fn element_at(&self, index: &BigInt) -> Self::E {
        self.element(index.clone())
    }
}

/// The order of the secp256k1 group, which is the modulus of its scalar field.
pub struct Secp256k1Order;

impl Prime for Secp256k1Order {
    const MODULUS: [u64; LIMBS] = [
        0xBFD2_5E8C_D036_4141,
        0xBAAE_DCE6_AF48_A03B,
        0xFFFF_FFFF_FFFF_FFFE,
        0xFFFF_FFFF_FFFF_FFFF,
    ];
}

/// The scalar field of secp256k1.
pub type Secp256k1Scalars = Fp<Secp256k1Order>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::elliptic::Curve;
    use crate::integers::{Integer, Integers};
    use crate::interpolation::interpolate;
    use crate::polynomials::RingExt;

    /// The base field of secp256k1, `2^256 - 2^32 - 977`, as a second modulus. Its
    /// least limb is `0x...FC2F`, where the scalar field's is `1` modulo 64, so the
    /// iteration that derives the inverse constant starts a round behind here.
    struct Secp256k1Base;

    impl Prime for Secp256k1Base {
        const MODULUS: [u64; LIMBS] = [
            0xFFFF_FFFE_FFFF_FC2F,
            0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF,
            0xFFFF_FFFF_FFFF_FFFF,
        ];
    }

    /// `value mod n`, in `0..n`, which `%` does not give for a negative value.
    fn modulo(value: BigInt, n: &BigInt) -> BigInt {
        let remainder = value % n;
        if remainder < BigInt::from(0u32) {
            remainder + n
        } else {
            remainder
        }
    }

    /// A spread of integers to test with: the edges, and some 256-bit values.
    fn values(n: &BigInt) -> Vec<BigInt> {
        let big = |text: &str| BigInt::parse_bytes(text.as_bytes(), 16).expect("hex");
        vec![
            BigInt::from(0u32),
            BigInt::from(1u32),
            BigInt::from(2u32),
            n - 1u32,
            n - 2u32,
            BigInt::from(-1),
            BigInt::from(u64::MAX),
            big("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"),
            big("1000000000000000000000000000000000000000000000000000000000000000"),
            big("7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"),
            big("BFD25E8CD0364140BAAEDCE6AF48A03BFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFF"),
            n * 3u32 + 7u32,
        ]
    }

    /// The two constants the modulus is supposed to determine, against the integers.
    fn check_the_constants<P: Prime>() {
        let n = Fp::<P>::new().modulus();

        // -n^{-1} mod 2^64.
        let radix = BigInt::from(1u32) << 64;
        let inverse = BigInt::from(P::INVERSE);
        assert_eq!(
            modulo(&inverse * &n, &radix),
            &radix - 1u32,
            "the inverse does not clear a limb"
        );

        // 2^512 mod n.
        assert_eq!(
            from_limbs(&P::RADIX_SQUARED),
            modulo(BigInt::from(1u32) << 512, &n),
            "the conversion constant is wrong"
        );
    }

    /// An integer, into Montgomery form and back.
    fn check_the_round_trip<P: Prime>() {
        let f = Fp::<P>::new();
        let n = f.modulus();
        for value in values(&n) {
            assert_eq!(
                f.element(value.clone()).to_integer(),
                modulo(value.clone(), &n),
                "{value} did not survive the round trip"
            );
        }
    }

    /// Every operation against the same thing over `BigInt`.
    fn check_the_arithmetic<P: Prime>() {
        let f = Fp::<P>::new();
        let n = f.modulus();
        // Extended Euclid over the integers, which is not how `invert` does it.
        let oracle = Integers::modulo(n.clone());

        for a in values(&n) {
            let x = f.element(a.clone());
            assert_eq!((-x).to_integer(), modulo(-a.clone(), &n), "-{a}");

            match f.invert(&x) {
                Some(inverse) => {
                    let expected = oracle
                        .invert(&oracle.element(Integer::from(a.clone())))
                        .expect("a nonzero element of a prime field is a unit");
                    let expected: BigInt = expected.representative().clone().into();
                    assert_eq!(inverse.to_integer(), modulo(expected, &n), "1/{a}");
                    assert_eq!((inverse * x).to_integer(), BigInt::from(1u32), "{a} / {a}");
                }
                None => assert!(f.is_zero(&x), "{a} has no inverse and is not zero"),
            }

            // Fermat, by a route the field does not take internally.
            assert_eq!(
                f.pow(&x, n.clone()).to_integer(),
                modulo(a.clone(), &n),
                "{a}^n"
            );

            for b in values(&n) {
                let y = f.element(b.clone());
                assert_eq!((x + y).to_integer(), modulo(&a + &b, &n), "{a} + {b}");
                assert_eq!((x - y).to_integer(), modulo(&a - &b, &n), "{a} - {b}");
                assert_eq!((x * y).to_integer(), modulo(&a * &b, &n), "{a} * {b}");

                let mut accumulated = x;
                accumulated *= &y;
                assert_eq!(accumulated.to_integer(), modulo(&a * &b, &n), "{a} *= {b}");
            }
        }
    }

    #[test]
    fn the_scalar_field_of_secp256k1() {
        // The modulus, against the hexadecimal the curve is published with.
        assert_eq!(
            Secp256k1Scalars::new().modulus(),
            BigInt::parse_bytes(
                b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
                16
            )
            .expect("hex")
        );
        check_the_constants::<Secp256k1Order>();
        check_the_round_trip::<Secp256k1Order>();
        check_the_arithmetic::<Secp256k1Order>();
    }

    #[test]
    fn a_second_modulus_of_another_shape() {
        assert_eq!(
            Fp::<Secp256k1Base>::new().modulus(),
            (BigInt::from(1u32) << 256) - (BigInt::from(1u32) << 32) - 977u32
        );
        check_the_constants::<Secp256k1Base>();
        check_the_round_trip::<Secp256k1Base>();
        check_the_arithmetic::<Secp256k1Base>();
    }

    #[test]
    fn the_generic_algorithms_take_it() {
        let f = Secp256k1Scalars::new();

        // Interpolation, which divides in the field.
        let points: Vec<_> = [1i64, 2, 3, 4].iter().map(|n| f.element(*n)).collect();
        let heights: Vec<_> = [5i64, 11, 29, 65].iter().map(|n| f.element(*n)).collect();
        let through =
            interpolate(&f.polynomials(), &points, &heights).expect("four distinct points");
        // x^3 + 4, which those four points lie on.
        for (point, height) in points.iter().zip(&heights) {
            assert_eq!(through.evaluate(point), *height);
        }
        assert_eq!(through.degree(), Some(3));

        // And the curve machinery, over a 256-bit field.
        let curve = Curve::new(f, f.element(0), f.element(7)).expect("nonsingular");
        let point = (1i64..20)
            .find_map(|n| {
                let x = f.element(n);
                f.sqrt(&curve.evaluate(&x)).and_then(|y| curve.point(x, y))
            })
            .expect("some small x is on y^2 = x^3 + 7");

        let doubled = point.clone() + point.clone();
        assert_eq!(curve.multiply(3, &point), doubled.clone() + point.clone());
        assert_eq!(
            curve.multiply(5, &point) + curve.multiply(7, &point),
            curve.multiply(12, &point)
        );
        assert_eq!(doubled + (-point.clone()), point);
    }
}

/// What the fixed modulus buys, measured rather than asserted. Run it with
/// `cargo test --release --lib timing -- --ignored --nocapture`.
#[cfg(test)]
mod timing {
    use super::*;
    use crate::integers::{Integer, Integers};
    use std::time::Instant;

    #[test]
    #[ignore = "a measurement, not an assertion, and it wants a release build"]
    fn against_arbitrary_precision() {
        let f = Secp256k1Scalars::new();
        let n = f.modulus();
        let zn = Integers::modulo(n.clone());
        let rounds = 200_000;

        let (a, b) = (f.element(&n / 3u32), f.element(&n / 7u32));
        let start = Instant::now();
        let mut x = a;
        for _ in 0..rounds {
            x *= b;
        }
        let fixed = start.elapsed();
        assert!(!f.is_zero(&x));

        let (c, d) = (
            zn.element(Integer::from(&n / 3u32)),
            zn.element(Integer::from(&n / 7u32)),
        );
        let start = Instant::now();
        let mut y = c;
        for _ in 0..rounds {
            y *= d.clone();
        }
        let general = start.elapsed();
        assert!(!CommutativeMonoid::is_zero(&zn, &y));

        println!(
            "multiply: fixed {:?}/op, BigInt {:?}/op, {:.1}x",
            fixed / rounds,
            general / rounds,
            general.as_secs_f64() / fixed.as_secs_f64()
        );

        let start = Instant::now();
        for _ in 0..1000 {
            let _ = f.invert(&a);
        }
        println!("invert (Fermat): {:?}/op", start.elapsed() / 1000);
    }
}
