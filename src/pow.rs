//! Exponentiation by square-and-multiply.

use crate::structures::Monoid;
use num_bigint::{BigInt, Sign};
use std::ops::{Mul, MulAssign};

/// `base^exp`, in `O(log exp)` multiplications. `exp == 0` gives the identity.
///
/// The exponent is any integer type, including [`BigInt`], so this is not capped at
/// a machine word.
///
/// # Panics
/// If `exp` is negative, which a monoid cannot express.
pub fn pow<M, T>(monoid: &M, base: &M::E, exp: T) -> M::E
where
    M: Monoid,
    M::E: Mul<Output = M::E> + for<'a> MulAssign<&'a M::E>,
    T: Into<BigInt>,
{
    let (sign, magnitude) = exp.into().into_parts();
    assert!(sign != Sign::Minus, "negative exponent in a monoid");

    let mut result = monoid.identity();
    let mut base = base.clone();
    let bits = magnitude.bits();
    for i in 0..bits {
        if magnitude.bit(i) {
            result *= &base;
        }
        // Skip the final squaring; nothing above the top bit will read it.
        if i + 1 < bits {
            base = base.clone() * base;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::{Integer, Integers};
    use crate::structures::Domain;
    use num_bigint::BigInt;

    #[test]
    fn squares_and_multiplies() {
        let z = Integers::default();
        assert_eq!(pow(&z, &Integer::from(3), 0), z.identity());
        assert_eq!(pow(&z, &Integer::from(3), 1), Integer::from(3));
        assert_eq!(pow(&z, &Integer::from(2), 10), Integer::from(1024));
        assert_eq!(pow(&z, &Integer::from(-3), 3), Integer::from(-27));

        // Against repeated multiplication, over exponents that exercise every
        // bit pattern up to 16.
        for exp in 0..16usize {
            let mut expected = z.identity();
            for _ in 0..exp {
                expected *= Integer::from(7);
            }
            assert_eq!(pow(&z, &Integer::from(7), exp), expected, "at exp = {exp}");
        }

        // Reduction happens in the ring, so this stays small: 2 has order 8 in F_17.
        let f17 = Integers::modulo(17);
        assert_eq!(pow(&f17, &f17.element(2), 8), f17.identity());
        assert_eq!(pow(&f17, &f17.element(2), 100), f17.element(16));
    }

    #[test]
    fn the_exponent_is_not_capped_at_a_machine_word() {
        // 2 has order 8 in F_17, so 2^k depends only on k mod 8 - which lets a
        // far-beyond-usize exponent be checked against a small one.
        let f17 = Integers::modulo(17);
        let two = f17.element(2);
        let huge = BigInt::from(2).pow(100);
        assert!(huge > BigInt::from(u64::MAX));

        // 2^100 is divisible by 8, so this is 2^0.
        assert_eq!(pow(&f17, &two, huge.clone()), f17.identity());
        assert_eq!(pow(&f17, &two, huge.clone() + 3), f17.element(8));
        assert_eq!(pow(&f17, &two, huge + 5), f17.element(15));

        // Small exponents still agree whichever integer type they arrive as.
        for exp in 0..20u32 {
            let expected = pow(&f17, &two, exp as usize);
            assert_eq!(pow(&f17, &two, BigInt::from(exp)), expected);
            assert_eq!(pow(&f17, &two, exp as i64), expected);
        }
    }

    #[test]
    #[should_panic(expected = "negative exponent")]
    fn rejects_a_negative_exponent() {
        let z = Integers::default();
        pow(&z, &Integer::from(2), -1i64);
    }
}
