//! The integers (`Z`) as a [`EuclideanDomain`] over [`BigInt`].

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, DivRem, EuclideanDomain, Monoid, QuotientRing, Ring,
    SemiRing, Semigroup, Set,
};
use derive_more::{Add, Display, From, Sub};
use num_bigint::BigInt;
use std::ops::Mul;

/// Handle for the set of integers; used to construct [`Integer`] values.
#[derive(Default, Clone, Debug)]
pub struct Integers {}

/// An integer, wrapping [`BigInt`].
#[derive(Display, From, Clone, PartialEq, Eq, Debug, Add, Sub)]
pub struct Integer(BigInt);

impl Set for Integers {
    type E = Integer;
}

impl Mul<Integer> for Integer {
    type Output = Integer;
    fn mul(self, rhs: Integer) -> Self::Output {
        Integer(self.0 * rhs.0)
    }
}

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
