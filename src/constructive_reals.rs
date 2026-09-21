//! Constructive (computable) real numbers, ported from Hans-J. Boehm's `CR`.
//!
//! A value is an expression tree evaluated only to the precision asked for:
//! `get_appr(p)` returns `a` with `|a · 2^p − value| < 2^p`.
//!
//! Equality is undecidable, so there is no `Eq`/`PartialEq` and comparison may answer
//! "undetermined". [`ConstructiveReals`] is a `Ring` but not a `Field`.

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, Domain, Monoid, Ring, SemiRing, Semigroup,
};
use num_bigint::{BigInt, Sign};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::ops::{Add, Mul, MulAssign, Neg, Sub};
use std::rc::Rc;

/// Returned by [`ConstructiveReal::msd`] when the value is too close to zero to locate.
const UNKNOWN_MSD: i32 = i32::MIN;

/// A constructive real. Cheap to clone: a refcounted handle onto a shared node.
#[derive(Clone)]
pub struct ConstructiveReal(Rc<Node>);

/// Handle for the constructive reals; the [`Ring`] impls hang off this.
#[derive(Default, Clone, Debug)]
pub struct ConstructiveReals;

macro_rules! constructive_real_from {
    ($($t:ty),*) => {$(
        impl From<$t> for ConstructiveReal {
            fn from(n: $t) -> Self {
                ConstructiveReal::from_int(n)
            }
        }
    )*};
}
constructive_real_from!(i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, BigInt);

impl Domain for ConstructiveReals {
    type E = ConstructiveReal;
    type Repr = ConstructiveReal;

    fn element<T: Into<ConstructiveReal>>(&self, value: T) -> ConstructiveReal {
        value.into()
    }
}

impl Semigroup for ConstructiveReals {}

impl Monoid for ConstructiveReals {
    fn identity(&self) -> ConstructiveReal {
        ConstructiveReal::from_int(1)
    }
}

impl CommutativeMonoid for ConstructiveReals {
    fn zero(&self) -> ConstructiveReal {
        ConstructiveReal::from_int(0)
    }
}

impl AdditiveGroup for ConstructiveReals {}
impl SemiRing for ConstructiveReals {}
impl Ring for ConstructiveReals {}

/// An expression node: the operation plus its best approximation so far.
struct Node {
    op: Op,
    cache: RefCell<Option<Approximation>>,
}

/// `max_appr` is `get_appr(min_prec)`; coarser precisions scale down from it.
struct Approximation {
    min_prec: i32,
    max_appr: BigInt,
}

/// The operation a node represents.
enum Op {
    /// An exact integer.
    Int(BigInt),
    /// Sum of two reals.
    Add(ConstructiveReal, ConstructiveReal),
    /// Negation.
    Neg(ConstructiveReal),
    /// `op · 2^count` (a left shift; `count` may be negative).
    Shift(ConstructiveReal, i32),
    /// Product of two reals.
    Mul(ConstructiveReal, ConstructiveReal),
    /// Non-negative square root.
    Sqrt(ConstructiveReal),
}

/// `true` iff `v` is zero.
fn is_zero(v: &BigInt) -> bool {
    v.sign() == Sign::NoSign
}

/// Absolute value of `v`.
fn abs(v: &BigInt) -> BigInt {
    if v.sign() == Sign::Minus {
        -v
    } else {
        v.clone()
    }
}

/// Shift `v` by `n` bits. Right shifts round half-up, matching Boehm's `scale`.
fn scale(v: BigInt, n: i32) -> BigInt {
    if n >= 0 {
        v << (n as usize)
    } else {
        let k = (-n) as usize;
        let adjusted = v + (BigInt::from(1) << (k - 1));
        adjusted >> k
    }
}

/// Floor square root by Newton's method, or zero for non-positive input.
fn isqrt(n: &BigInt) -> BigInt {
    if n.sign() != Sign::Plus {
        return BigInt::from(0);
    }
    // Start above the true root so the iteration decreases monotonically.
    let mut guess = BigInt::from(1) << (n.bits() as usize).div_ceil(2);
    loop {
        let next = (&guess + n / &guess) >> 1usize;
        if next >= guess {
            return guess;
        }
        guess = next;
    }
}

impl ConstructiveReal {
    /// Wrap an operation in a fresh, uncached node.
    fn new(op: Op) -> ConstructiveReal {
        ConstructiveReal(Rc::new(Node {
            op,
            cache: RefCell::new(None),
        }))
    }

    /// The constructive real exactly equal to the given integer.
    pub fn from_int(n: impl Into<BigInt>) -> ConstructiveReal {
        ConstructiveReal::new(Op::Int(n.into()))
    }

    /// `self · 2^n`.
    pub fn shift_left(&self, n: i32) -> ConstructiveReal {
        ConstructiveReal::new(Op::Shift(self.clone(), n))
    }

    /// `self / 2^n`.
    pub fn shift_right(&self, n: i32) -> ConstructiveReal {
        ConstructiveReal::new(Op::Shift(self.clone(), -n))
    }

    /// The non-negative square root. A negative operand is unspecified rather than an
    /// error, since negativity is not decidable.
    pub fn sqrt(&self) -> ConstructiveReal {
        ConstructiveReal::new(Op::Sqrt(self.clone()))
    }

    /// An `a` with `|a · 2^precision − value| < 2^precision`, reusing the cache.
    fn get_appr(&self, precision: i32) -> BigInt {
        {
            let cache = self.0.cache.borrow();
            if let Some(appr) = cache.as_ref() {
                if precision >= appr.min_prec {
                    return scale(appr.max_appr.clone(), appr.min_prec - precision);
                }
            }
        }
        let result = self.approximate(precision);
        *self.0.cache.borrow_mut() = Some(Approximation {
            min_prec: precision,
            max_appr: result.clone(),
        });
        result
    }

    /// Compute a fresh approximation at `precision` directly from the operation.
    fn approximate(&self, p: i32) -> BigInt {
        match &self.0.op {
            Op::Int(value) => scale(value.clone(), -p),
            Op::Add(a, b) => scale(a.get_appr(p - 2) + b.get_appr(p - 2), -2),
            Op::Neg(a) => -a.get_appr(p),
            Op::Shift(a, count) => a.get_appr(p - count),
            Op::Mul(a, b) => approximate_mul(a, b, p),
            Op::Sqrt(a) => {
                let working = p - 3;
                let operand = a.get_appr(2 * working);
                match operand.sign() {
                    // Operand assumed non-negative: a non-positive approximation
                    // means it is indistinguishable from zero at this precision.
                    Sign::Minus | Sign::NoSign => BigInt::from(0),
                    Sign::Plus => scale(isqrt(&operand), -3),
                }
            }
        }
    }

    /// Most significant digit position. Requires a populated cache.
    fn known_msd(&self) -> i32 {
        let cache = self.0.cache.borrow();
        let appr = cache
            .as_ref()
            .expect("known_msd called before any approximation");
        let length = appr.max_appr.bits() as i32;
        appr.min_prec + length - 1
    }

    /// Most significant digit, refining to `n`. [`UNKNOWN_MSD`] if indistinguishable
    /// from zero there.
    fn msd(&self, n: i32) -> i32 {
        let needs_refine = match self.0.cache.borrow().as_ref() {
            None => true,
            Some(appr) => abs(&appr.max_appr) <= BigInt::from(1),
        };
        if needs_refine {
            self.get_appr(n - 1);
            let cache = self.0.cache.borrow();
            let appr = cache.as_ref().unwrap();
            if abs(&appr.max_appr) <= BigInt::from(1) {
                return UNKNOWN_MSD;
            }
        }
        self.known_msd()
    }

    /// Compare at one precision; `None` when the values are too close to tell.
    pub fn compare_at(&self, other: &ConstructiveReal, precision: i32) -> Option<Ordering> {
        let needed = precision - 1;
        let this_appr = self.get_appr(needed);
        let other_appr = other.get_appr(needed);
        let one = BigInt::from(1);
        if this_appr > &other_appr + &one {
            Some(Ordering::Greater)
        } else if this_appr < &other_appr - &one {
            Some(Ordering::Less)
        } else {
            None
        }
    }

    /// Compare, refining from `-20` until resolvable or past `min_precision`. `None` if
    /// unresolved: equality is undecidable, so the bound is what guarantees termination.
    pub fn compare(&self, other: &ConstructiveReal, min_precision: i32) -> Option<Ordering> {
        let mut a = -20;
        loop {
            if let Some(ordering) = self.compare_at(other, a) {
                return Some(ordering);
            }
            if a <= min_precision {
                return None;
            }
            a = a.saturating_mul(2);
        }
    }

    /// The sign (`-1`, `0`, `1`) if resolvable at `precision`. A nonzero answer is
    /// always correct.
    pub fn sign_at(&self, precision: i32) -> Option<i32> {
        match self.get_appr(precision - 1).sign() {
            Sign::Minus => Some(-1),
            Sign::Plus => Some(1),
            Sign::NoSign => None,
        }
    }

    /// Render with `digits` digits after the point. The last digit may be off by one
    /// for non-dyadic values.
    pub fn to_decimal(&self, digits: u32) -> String {
        let ten = BigInt::from(10);
        let mut scale_factor = BigInt::from(1);
        for _ in 0..digits {
            scale_factor *= &ten;
        }
        let scaled = self.clone() * ConstructiveReal::from_int(scale_factor);
        let mut value = scaled.get_appr(0);

        let negative = value.sign() == Sign::Minus;
        if negative {
            value = -value;
        }
        let mut text = value.to_str_radix(10);
        let d = digits as usize;
        if d == 0 {
            return if negative { format!("-{}", text) } else { text };
        }
        if text.len() <= d {
            text = "0".repeat(d + 1 - text.len()) + &text;
        }
        let point = text.len() - d;
        let body = format!("{}.{}", &text[..point], &text[point..]);
        if negative {
            format!("-{}", body)
        } else {
            body
        }
    }
}

/// Approximate a product, following Boehm's `mult_CR`: each factor at a reduced
/// precision sized by the other's msd.
fn approximate_mul(op1: &ConstructiveReal, op2: &ConstructiveReal, p: i32) -> BigInt {
    let half_prec = (p >> 1) - 1;

    // Arrange so `larger` is the operand whose magnitude we have pinned down.
    let mut larger = op1.clone();
    let mut smaller = op2.clone();
    let mut msd_larger = larger.msd(half_prec);
    if msd_larger == UNKNOWN_MSD {
        let msd_other = smaller.msd(half_prec);
        if msd_other == UNKNOWN_MSD {
            // Both factors are tiny; their product is small enough that zero
            // satisfies the contract at precision `p`.
            return BigInt::from(0);
        }
        std::mem::swap(&mut larger, &mut smaller);
        msd_larger = msd_other;
    }

    let prec_smaller = p - msd_larger - 3;
    let appr_smaller = smaller.get_appr(prec_smaller);
    if is_zero(&appr_smaller) {
        return BigInt::from(0);
    }
    let msd_smaller = smaller.known_msd();
    let prec_larger = p - msd_smaller - 3;
    let appr_larger = larger.get_appr(prec_larger);

    let scale_digits = prec_larger + prec_smaller - p;
    scale(appr_larger * appr_smaller, scale_digits)
}

impl Add for ConstructiveReal {
    type Output = ConstructiveReal;
    fn add(self, rhs: ConstructiveReal) -> ConstructiveReal {
        ConstructiveReal::new(Op::Add(self, rhs))
    }
}

impl Neg for ConstructiveReal {
    type Output = ConstructiveReal;
    fn neg(self) -> ConstructiveReal {
        ConstructiveReal::new(Op::Neg(self))
    }
}

impl Sub for ConstructiveReal {
    type Output = ConstructiveReal;
    fn sub(self, rhs: ConstructiveReal) -> ConstructiveReal {
        ConstructiveReal::new(Op::Add(self, ConstructiveReal::new(Op::Neg(rhs))))
    }
}

impl Mul for ConstructiveReal {
    type Output = ConstructiveReal;
    fn mul(self, rhs: ConstructiveReal) -> ConstructiveReal {
        ConstructiveReal::new(Op::Mul(self, rhs))
    }
}

impl MulAssign<&ConstructiveReal> for ConstructiveReal {
    fn mul_assign(&mut self, rhs: &ConstructiveReal) {
        *self = ConstructiveReal::new(Op::Mul(self.clone(), rhs.clone()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_constants_render() {
        assert_eq!(ConstructiveReal::from_int(7).to_decimal(3), "7.000");
        assert_eq!(ConstructiveReal::from_int(-5).to_decimal(2), "-5.00");
        assert_eq!(ConstructiveReal::from_int(0).to_decimal(0), "0");
        assert_eq!(ConstructiveReal::from_int(42).to_decimal(0), "42");
    }

    #[test]
    fn addition_and_subtraction() {
        let sum = ConstructiveReal::from_int(1) + ConstructiveReal::from_int(2);
        assert_eq!(sum.to_decimal(0), "3");

        let diff = ConstructiveReal::from_int(10) - ConstructiveReal::from_int(25);
        assert_eq!(diff.to_decimal(1), "-15.0");

        let neg = -ConstructiveReal::from_int(8);
        assert_eq!(neg.to_decimal(0), "-8");
    }

    #[test]
    fn multiplication() {
        let product = ConstructiveReal::from_int(6) * ConstructiveReal::from_int(7);
        assert_eq!(product.to_decimal(0), "42");

        let negative = ConstructiveReal::from_int(-3) * ConstructiveReal::from_int(4);
        assert_eq!(negative.to_decimal(2), "-12.00");

        // Multiplying by zero short-circuits in approximate_mul.
        let zero = ConstructiveReal::from_int(0) * ConstructiveReal::from_int(999);
        assert_eq!(zero.to_decimal(0), "0");
    }
}
