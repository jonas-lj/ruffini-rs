//! Exponentiation by square-and-multiply.

use crate::structures::Monoid;
use std::ops::Mul;

/// `base^exp`, in `O(log exp)` multiplications. `exp == 0` gives the identity.
pub fn pow<M>(monoid: &M, base: &M::E, mut exp: usize) -> M::E
where
    M: Monoid,
    M::E: Mul<Output = M::E>,
{
    let mut result = monoid.identity();
    let mut base = base.clone();
    while exp > 0 {
        if exp & 1 == 1 {
            result = result * base.clone();
        }
        exp >>= 1;
        if exp > 0 {
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
                expected = expected * Integer::from(7);
            }
            assert_eq!(pow(&z, &Integer::from(7), exp), expected, "at exp = {exp}");
        }

        // Reduction happens in the ring, so this stays small: 2 has order 8 in F_17.
        let f17 = Integers::modulo(17);
        assert_eq!(pow(&f17, &f17.element(2), 8), f17.identity());
        assert_eq!(pow(&f17, &f17.element(2), 100), f17.element(16));
    }
}
