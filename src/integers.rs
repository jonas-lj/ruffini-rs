//! The integers (`Z`) as a [`EuclideanDomain`] over [`BigInt`].

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, DivRem, EuclideanDomain, Monoid, QuotientRing, Ring,
    SemiRing, Semigroup, Domain,
};
use derive_more::{Add, Display, From, Sub};
use num_bigint::BigInt;
use std::ops::{AddAssign, Mul, MulAssign, SubAssign};
use std::rc::Rc;

/// Handle for the set of integers; used to construct [`Integer`] values.
#[derive(Default, Clone, Debug)]
pub struct Integers {}

/// An integer, wrapping [`BigInt`].
#[derive(Display, From, Clone, PartialEq, Eq, Debug, Add, Sub)]
#[from(BigInt, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize)]
pub struct Integer(BigInt);

impl Integers {
    /// The ring of integers modulo `modulus`, e.g. `Integers::modulo(7)`. A prime
    /// modulus gives the finite field `F_p`.
    pub fn modulo(modulus: impl Into<Integer>) -> Rc<QuotientRing<Integers>> {
        QuotientRing::new(Integers::default(), modulus.into())
    }
}

impl Domain for Integers {
    type E = Integer;
    type Repr = Integer;

    fn element<T: Into<Integer>>(&self, value: T) -> Integer {
        value.into()
    }
}

impl Mul<Integer> for Integer {
    type Output = Integer;
    fn mul(self, rhs: Integer) -> Self::Output {
        Integer(self.0 * rhs.0)
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

/// Borrowed operand combinations, forwarded to [`BigInt`]'s own reference ops so that
/// `&a + &b` allocates the result without copying either input.
macro_rules! integer_ref_ops {
    ($($op:ident, $method:ident);* $(;)?) => {$(
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
integer_ref_ops!(Add, add; Sub, sub; Mul, mul);

impl DivRem for Integer {
    fn div_rem(&self, divisor: &Self) -> (Self, Self) {
        let q = &self.0 / &divisor.0;
        let r = &self.0 % &divisor.0;
        (Integer(q), Integer(r))
    }
}

impl From<Integer> for BigInt {
    fn from(value: Integer) -> Self {
        value.0
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
