//! The integers (`Z`) as a [`EuclideanDomain`] over [`BigInt`].

use crate::fractions::FractionField;
use crate::structures::{
    AdditiveGroup, CommutativeMonoid, DivRem, EuclideanDomain, Monoid, QuotientRing, Ring,
    SemiRing, Semigroup, Domain,
};
use num_bigint::BigInt;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
use std::sync::Arc;

/// Handle for the set of integers, used to construct [`Integer`] values.
#[derive(Default, Clone, Debug)]
pub struct Integers {}

/// An integer, wrapping [`BigInt`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Integer(BigInt);

impl fmt::Display for Integer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl From<Integer> for BigInt {
    fn from(n: Integer) -> BigInt {
        n.0
    }
}

/// `From` for each integer type [`BigInt`] accepts, so that a literal can stand for an
/// element wherever one is asked for.
///
/// Listed rather than blanket over `T: Into<BigInt>`: that would collide with the
/// reflexive `From<Integer> for Integer`, since `From<Integer> for BigInt` exists.
macro_rules! integer_from {
    ($($from:ty),+ $(,)?) => {$(
        impl From<$from> for Integer {
            fn from(n: $from) -> Integer {
                Integer(BigInt::from(n))
            }
        }
    )+};
}
integer_from!(BigInt, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize);

impl Integers {
    /// The ring of integers modulo `modulus`, e.g. `Integers::modulo(7)`. A prime
    /// modulus gives the finite field `F_p`.
    pub fn modulo(modulus: impl Into<Integer>) -> Arc<QuotientRing<Integers>> {
        Integers::default().quotient(modulus.into())
    }

    /// The rationals `Q`, the field of fractions of `Z`.
    pub fn rationals() -> Arc<FractionField<Integers>> {
        Integers::default().fractions()
    }
}

impl Domain for Integers {
    type E = Integer;
    type Repr = Integer;

    fn element<T: Into<Integer>>(&self, value: T) -> Integer {
        value.into()
    }
}

/// Compound assignment, forwarded to [`BigInt`]'s own, so nothing is cloned.
macro_rules! integer_assign_ops {
    ($($op:ident, $method:ident);* $(;)?) => {$(
        impl $op<&Integer> for Integer {
            fn $method(&mut self, rhs: &Integer) {
                self.0.$method(&rhs.0);
            }
        }

        impl $op<Integer> for Integer {
            fn $method(&mut self, rhs: Integer) {
                self.0.$method(rhs.0);
            }
        }
    )*};
}
integer_assign_ops!(AddAssign, add_assign; SubAssign, sub_assign; MulAssign, mul_assign);

/// Every operand combination, forwarded to [`BigInt`]'s own, so that `&a + &b`
/// allocates the result without copying either input.
macro_rules! integer_ops {
    ($($op:ident, $method:ident);* $(;)?) => {$(
        impl $op for Integer {
            type Output = Integer;
            fn $method(self, rhs: Integer) -> Integer {
                Integer(self.0.$method(rhs.0))
            }
        }

        impl $op<&Integer> for &Integer {
            type Output = Integer;
            fn $method(self, rhs: &Integer) -> Integer {
                Integer((&self.0).$method(&rhs.0))
            }
        }

        impl $op<&Integer> for Integer {
            type Output = Integer;
            fn $method(self, rhs: &Integer) -> Integer {
                Integer(self.0.$method(&rhs.0))
            }
        }

        impl $op<Integer> for &Integer {
            type Output = Integer;
            fn $method(self, rhs: Integer) -> Integer {
                Integer((&self.0).$method(rhs.0))
            }
        }
    )*};
}
integer_ops!(Add, add; Sub, sub; Mul, mul);

impl Neg for Integer {
    type Output = Integer;
    fn neg(self) -> Integer {
        Integer(-self.0)
    }
}

impl DivRem for Integer {
    fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let q = &self.0 / &divisor.0;
        let r = &self.0 % &divisor.0;
        (Integer(q), Integer(r))
    }
}

impl Semigroup for Integers {}

impl Monoid for Integers {
    fn identity(&self) -> Self::E {
        Integer(BigInt::from(1))
    }
}

impl CommutativeMonoid for Integers {
    fn zero(&self) -> Self::E {
        Integer(BigInt::from(0))
    }
}

impl AdditiveGroup for Integers {}
impl SemiRing for Integers {}
impl Ring for Integers {}

impl EuclideanDomain for Integers {
    /// The sign of `x` (`-1` for negative, `1` for non-negative including zero).
    fn unit_part(&self, x: &Self::E) -> Self::E {
        if x.0 < BigInt::from(0) {
            Integer(BigInt::from(-1))
        } else {
            Integer(BigInt::from(1))
        }
    }

    /// `±1` are self-inverse. The result for non-unit inputs is meaningless
    /// (the trait only requires correctness on units).
    fn unit_inverse(&self, u: &Self::E) -> Self::E {
        u.clone()
    }
}

impl QuotientRing<Integers> {
    /// The number of elements in `Z/(modulus)`, i.e. `|modulus|`.
    ///
    /// When the modulus is prime this is the order of the finite field `F_p`.
    pub fn order(&self) -> BigInt {
        let m: BigInt = self.modulus().clone().into();
        if m < BigInt::from(0) {
            -m
        } else {
            m
        }
    }
}
