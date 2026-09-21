//! Ruffini: algebraic structures over arbitrary-precision integers.
//!
//! Provides traits for the algebraic hierarchy
//! ([`Domain`](structures::Domain) → [`Semigroup`](structures::Semigroup) → …
//! → [`EuclideanDomain`](structures::EuclideanDomain) → [`Field`](structures::Field))
//! along with concrete implementations: the [integers], generic
//! [`QuotientRing`](structures::QuotientRing) and [`Polynomial`](polynomials::Polynomial)
//! constructions. A finite prime field `F_p` is just `Rc<QuotientRing<Integers>>`
//! with a prime modulus; `F_p[x]` is then `Rc<PolynomialRing<Rc<QuotientRing<Integers>>>>`.
//!
//! [`Domain::E`](structures::Domain::E) does not require `Eq`; only
//! [`EuclideanDomain`](structures::EuclideanDomain) and [`Field`](structures::Field) do.

/// Compiles the README's examples as doctests, so they cannot go stale.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

pub mod constructive_reals;
pub mod integers;
pub mod polynomials;
pub mod structures;

#[cfg(test)]
mod tests {
    use crate::integers::{Integer, Integers};
    use crate::polynomials::PolynomialRing;
    use crate::structures::{
        CommutativeMonoid, DivRem, Domain, EuclideanDomain, Field, Monoid, Ring,
    };
    use num_bigint::BigInt;

    fn int(n: i64) -> Integer {
        Integer::from(n)
    }

    #[test]
    fn integers_add() {
        assert_eq!(format!("{}", int(1) + int(2)), "3");
    }

    #[test]
    fn integers_div_rem() {
        let (q, r) = int(17).div_rem(&int(5));
        assert_eq!(format!("{}", q), "3");
        assert_eq!(format!("{}", r), "2");
    }

    #[test]
    fn quotient_ring_arithmetic() {
        let f7 = Integers::modulo(7);
        let three = f7.element(3);
        let six = f7.element(6);
        // 3 + 6 = 9 ≡ 2 (mod 7)
        assert_eq!(&three + &six, f7.element(2));
        // 3 * 6 = 18 ≡ 4 (mod 7)
        assert_eq!(&three * &six, f7.element(4));
        // zero and identity
        assert_eq!(f7.zero(), f7.element(0));
        assert_eq!(f7.identity(), f7.element(1));

        // `eq` compares plain integers as representatives, without naming elements
        assert!(f7.eq(3 + 6, 2));
        assert!(!f7.eq(3 + 6, 3));
        assert!(f7.eq(-1, 6));
    }

    #[test]
    fn prime_field_f7() {
        let f7 = Integers::modulo(7);

        // 3 * 5 = 15 ≡ 1 (mod 7), so 3^{-1} = 5
        assert_eq!(f7.inverse(3), Some(f7.element(5)));
        // every nonzero element has an inverse that multiplies back to 1
        for n in 1..7 {
            let x = f7.element(n);
            let inv = f7.invert(&x).expect("nonzero element must have an inverse");
            assert_eq!(&x * &inv, f7.identity());
        }
        // zero has no inverse
        assert_eq!(f7.inverse(0), None);
        // order
        assert_eq!(f7.order(), BigInt::from(7));
    }

    #[test]
    fn extended_gcd_canonicalises_via_unit_part() {
        let z = Integers::default();
        // gcd(4, 7) = 1, regardless of sign of inputs — canonical gcd is positive.
        let (g, s, t) = z.extended_gcd(int(4), int(7));
        assert_eq!(g, int(1));
        assert_eq!(&s * &int(4) + &t * &int(7), int(1));

        // With a negative input, the raw Euclidean step would land on gcd = -1;
        // canonicalisation should still produce gcd = 1.
        let (g, s, t) = z.extended_gcd(int(-4), int(7));
        assert_eq!(g, int(1));
        assert_eq!(&s * &int(-4) + &t * &int(7), int(1));

        // Non-coprime: gcd(6, 10) = 2.
        let (g, _, _) = z.extended_gcd(int(6), int(10));
        assert_eq!(g, int(2));
    }

    #[test]
    fn polynomial_ring_z_x() {
        let zx = PolynomialRing::new(Integers::default());
        let poly = |coeffs: Vec<i64>| zx.element(coeffs.into_iter().map(int).collect::<Vec<_>>());

        let p = poly(vec![1, 2, 3]); // 1 + 2x + 3x^2
        let q = poly(vec![4, 5]);    // 4 + 5x

        // p + q = 5 + 7x + 3x^2
        assert_eq!(&p + &q, poly(vec![5, 7, 3]));
        // p - q = -3 - 3x + 3x^2
        assert_eq!(&p - &q, poly(vec![-3, -3, 3]));
        // p * q = 4 + 13x + 22x^2 + 15x^3
        assert_eq!(&p * &q, poly(vec![4, 13, 22, 15]));

        // trim trailing zeros
        assert_eq!(poly(vec![1, 2, 0, 0]), poly(vec![1, 2]));
        // zero polynomial and identity
        assert_eq!(zx.zero(), poly(vec![]));
        assert_eq!(zx.zero().degree(), None);
        assert_eq!(zx.identity(), poly(vec![1]));
        assert_eq!(p.degree(), Some(2));
        // Display
        assert_eq!(format!("{}", poly(vec![1, 2, 3])), "1 + 2*x + 3*x^{2}");
    }

    #[test]
    fn polynomial_division_in_f7_x() {
        let f7 = Integers::modulo(7);
        let f7x = PolynomialRing::new(f7.clone());
        let poly = |coeffs: Vec<i64>| {
            f7x.element(coeffs.into_iter().map(|n| f7.element(n)).collect::<Vec<_>>())
        };

        // (1 + x^2) divided by (2 + x) in F_7[x]:
        //   q = 5 + x,  r = 5     (verify: (5+x)(2+x) + 5 = 1 + x^2 mod 7)
        let p = poly(vec![1, 0, 1]);
        let d = poly(vec![2, 1]);
        let (q, r) = p.div_rem(&d);
        assert_eq!(q, poly(vec![5, 1]));
        assert_eq!(r, poly(vec![5]));
        assert_eq!(&q * &d + &r, p);

        // Exact division: (x + 1)(x + 2) = x^2 + 3x + 2  =>  divide by (x + 1) yields (x + 2) rem 0
        let p = poly(vec![2, 3, 1]);
        let d = poly(vec![1, 1]);
        let (q, r) = p.div_rem(&d);
        assert_eq!(q, poly(vec![2, 1]));
        assert_eq!(r, f7x.zero());
    }

    #[test]
    fn polynomial_gcd_in_f7_x_is_monic() {
        let f7 = Integers::modulo(7);
        let f7x = PolynomialRing::new(f7.clone());
        let poly = |coeffs: Vec<i64>| {
            f7x.element(coeffs.into_iter().map(|n| f7.element(n)).collect::<Vec<_>>())
        };

        // gcd(x^2 - 1, x - 1) = x - 1 in F_7[x]
        // x^2 - 1 ≡ [6, 0, 1]; x - 1 ≡ [6, 1]; gcd should be [6, 1] (monic).
        let a = poly(vec![6, 0, 1]);
        let b = poly(vec![6, 1]);
        let (g, s, t) = f7x.extended_gcd(a.clone(), b.clone());
        assert_eq!(g, poly(vec![6, 1]));
        // Bezout: s*a + t*b == g
        assert_eq!(&s * &a + &t * &b, g);

        // gcd(x^2 - 1, 2x - 2) — the raw Euclidean step produces gcd = 2x - 2
        // with leading coefficient 2; canonicalisation should make it monic (x - 1).
        // x^2 - 1 ≡ [6, 0, 1]; 2x - 2 ≡ [5, 2]; canonical gcd should be [6, 1].
        let a = poly(vec![6, 0, 1]);
        let b = poly(vec![5, 2]);
        let (g, s, t) = f7x.extended_gcd(a.clone(), b.clone());
        assert_eq!(g, poly(vec![6, 1])); // x - 1
        assert_eq!(&s * &a + &t * &b, g);
    }

    #[test]
    fn polynomial_over_f7() {
        let f7 = Integers::modulo(7);
        let f7x = PolynomialRing::new(f7.clone());
        let poly = |coeffs: Vec<i64>| {
            f7x.element(coeffs.into_iter().map(|n| f7.element(n)).collect::<Vec<_>>())
        };

        // (1 + 2x)(3 + 4x) = 3 + 10x + 8x^2  ≡  3 + 3x + x^2 (mod 7)
        assert_eq!(poly(vec![1, 2]) * poly(vec![3, 4]), poly(vec![3, 3, 1]));

        // (1 + x)(6 + x) = 6 + 7x + x^2 ≡ 6 + x^2 (mod 7)  — the 7x term vanishes
        // and the resulting middle zero coefficient stays (it's not trailing).
        assert_eq!(poly(vec![1, 1]) * poly(vec![6, 1]), poly(vec![6, 0, 1]));
    }

    #[test]
    fn composite_modulus_no_inverse_for_zero_divisors() {
        // Z/6Z is not a field. inverse(2) and inverse(3) should return None
        // (they are zero divisors); inverse(5) should still work.
        let zmod6 = Integers::modulo(6);
        assert_eq!(zmod6.inverse(2), None);
        assert_eq!(zmod6.inverse(3), None);
        // 5*5 = 25 ≡ 1 (mod 6)
        assert_eq!(zmod6.inverse(5), Some(zmod6.element(5)));
    }

    #[test]
    fn arithmetic_against_plain_integers() {
        let f7 = Integers::modulo(7);
        // 3 · 3⁻¹ = 1 in F_7, with neither side spelled out as an element.
        assert_eq!(f7.inverse(3).unwrap() * 3, f7.identity());
        // 3 + 6 = 9 ≡ 2, 3 - 6 = -3 ≡ 4, 3 * 6 = 18 ≡ 4 (mod 7)
        assert_eq!(f7.element(3) + 6, f7.element(2));
        assert_eq!(f7.element(3) - 6, f7.element(4));
        assert_eq!(&f7.element(3) * 6, f7.element(4));
        // The integer is reduced into the ring first, so it may be arbitrarily large.
        assert_eq!(f7.element(1) * BigInt::from(15), f7.element(1));

        // Same for polynomials, where the integer becomes a constant polynomial.
        let zx = PolynomialRing::new(Integers::default());
        let p = zx.element(vec![int(1), int(2)]); // 1 + 2x
        assert_eq!(&p * 3, zx.element(vec![int(3), int(6)]));
        assert_eq!(&p + 4, zx.element(vec![int(5), int(2)]));
        assert_eq!(p - 1, zx.element(vec![int(0), int(2)]));
    }

    #[test]
    fn from_integer_agrees_with_direct_construction() {
        let z = Integers::default();
        // Small values, including zero and negatives.
        for n in -20i64..=20 {
            assert_eq!(z.from_integer(n), int(n));
        }
        // Powers of two and their neighbours exercise the bit loop's boundaries:
        // 2^k sets exactly the top bit, 2^k - 1 sets every bit below it.
        for k in 0..62u32 {
            let p = 1i64 << k;
            assert_eq!(z.from_integer(p), int(p));
            assert_eq!(z.from_integer(p - 1), int(p - 1));
            assert_eq!(z.from_integer(-p), int(-p));
        }
    }

    /// A ring whose elements have no decidable equality. If this module compiles, the
    /// hierarchy up to `Ring` really does not demand `Eq`.
    mod without_eq {
        use crate::structures::{
            AdditiveGroup, CommutativeMonoid, Domain, Monoid, Ring, SemiRing, Semigroup,
        };
        use std::ops::{Add, Mul, Sub};

        /// Deliberately implements neither `PartialEq` nor `Eq`.
        #[derive(Clone)]
        struct Opaque(f64);

        impl Add for Opaque {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                Opaque(self.0 + rhs.0)
            }
        }
        impl Sub for Opaque {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                Opaque(self.0 - rhs.0)
            }
        }
        impl Mul for Opaque {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                Opaque(self.0 * rhs.0)
            }
        }

        #[derive(Clone)]
        struct Opaques;

        impl Domain for Opaques {
            type E = Opaque;
            type Repr = Opaque;
            fn element<T: Into<Opaque>>(&self, value: T) -> Opaque {
                value.into()
            }
        }
        impl Semigroup for Opaques {}
        impl Monoid for Opaques {
            fn identity(&self) -> Opaque {
                Opaque(1.0)
            }
        }
        impl CommutativeMonoid for Opaques {
            fn zero(&self) -> Opaque {
                Opaque(0.0)
            }
        }
        impl AdditiveGroup for Opaques {}
        impl SemiRing for Opaques {}
        impl Ring for Opaques {}

        #[test]
        fn reaches_ring_without_a_decidable_equality() {
            let r = Opaques;
            let sum = r.element(Opaque(2.0)) + r.identity();
            assert!((sum.0 - 3.0).abs() < 1e-12);
            let product = r.element(Opaque(3.0)) * r.element(Opaque(4.0));
            assert!((product.0 - 12.0).abs() < 1e-12);
        }
    }
}
