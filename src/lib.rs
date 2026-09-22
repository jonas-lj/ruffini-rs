//! Ruffini: algebraic structures over arbitrary-precision integers.
//!
//! Provides traits for the algebraic hierarchy
//! ([`Domain`](structures::Domain) → [`Semigroup`](structures::Semigroup) → …
//! → [`EuclideanDomain`](structures::EuclideanDomain) → [`Field`](structures::Field))
//! along with concrete implementations: the [integers], generic
//! [`QuotientRing`](structures::QuotientRing) and [`Polynomial`](polynomials::Polynomial)
//! constructions. A finite prime field `F_p` is just `Rc<QuotientRing<Integers>>`
//! with a prime modulus. `F_p[x]` is then `Rc<PolynomialRing<Rc<QuotientRing<Integers>>>>`.
//!
//! [`Domain::E`](structures::Domain::E) does not require `Eq`. Only
//! [`EuclideanDomain`](structures::EuclideanDomain) and [`Field`](structures::Field) do.

/// Compiles the README's examples as doctests, so they cannot go stale.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
pub struct ReadmeDoctests;

#[macro_use]
mod macros;

pub mod constructive_reals;
pub mod euclidean;
pub mod fft;
pub mod integers;
pub mod matrices;
pub mod multivariate;
pub mod polynomials;
pub mod pow;
pub mod structures;

#[cfg(test)]
mod tests {
    use crate::integers::{Integer, Integers};
    use crate::polynomials::RingExt;
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
    fn polynomial_ring_z_x() {
        let zx = Integers::default().polynomials();
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
        // The leading coefficient; the zero polynomial has none.
        assert_eq!(p.lead(), Some(&int(3)));
        assert_eq!(poly(vec![1, 2, 0, 0]).lead(), Some(&int(2)));
        assert_eq!(zx.zero().lead(), None);
        // Display
        assert_eq!(format!("{}", poly(vec![1, 2, 3])), "1 + 2*x + 3*x^{2}");

        // The constructor composes: Z[x][y].
        let zxy = zx.polynomials();
        assert_eq!(zxy.element(vec![p.clone(), zx.identity()]).degree(), Some(1));
    }

    #[test]
    fn polynomial_product_matches_schoolbook_for_every_shape() {
        let zx = Integers::default().polynomials();
        let poly = |c: &[i64]| zx.element(c.iter().map(|n| int(*n)).collect::<Vec<_>>());

        // Scatter-style reference, against the gathered implementation.
        fn reference(a: &[i64], b: &[i64]) -> Vec<i64> {
            if a.is_empty() || b.is_empty() {
                return Vec::new();
            }
            let mut out = vec![0i64; a.len() + b.len() - 1];
            for (i, x) in a.iter().enumerate() {
                for (j, y) in b.iter().enumerate() {
                    out[i + j] += x * y;
                }
            }
            while out.last() == Some(&0) {
                out.pop();
            }
            out
        }

        // Asymmetric lengths are where the gather bounds bite.
        for n in 0..6usize {
            for m in 0..6usize {
                let a: Vec<i64> = (1..=n as i64).map(|i| i - 3).collect();
                let b: Vec<i64> = (1..=m as i64).map(|j| 2 * j - 5).collect();
                assert_eq!(
                    poly(&a) * poly(&b),
                    poly(&reference(&a, &b)),
                    "n = {n}, m = {m}"
                );
                // Borrowed operator takes the same path.
                assert_eq!(&poly(&a) * &poly(&b), poly(&reference(&a, &b)));
            }
        }
    }

    #[test]
    fn polynomial_evaluation_by_horner() {
        let zx = Integers::default().polynomials();
        let poly = |c: &[i64]| zx.element(c.iter().map(|n| int(*n)).collect::<Vec<_>>());

        // Against direct evaluation of sum c_i x^i, over several shapes and points.
        for coeffs in [
            vec![],
            vec![7],
            vec![1, 2],
            vec![1, 0, -3, 4],
            vec![-5, 2, 0, 0, 1],
        ] {
            for x in [-3i64, -1, 0, 1, 2, 10] {
                let expected: i64 = coeffs
                    .iter()
                    .enumerate()
                    .map(|(i, c)| c * x.pow(i as u32))
                    .sum();
                assert_eq!(
                    poly(&coeffs).evaluate(&int(x)),
                    int(expected),
                    "coeffs = {coeffs:?}, x = {x}"
                );
            }
        }

        // The zero polynomial is zero everywhere; a constant ignores x.
        assert_eq!(poly(&[]).evaluate(&int(9)), zx.coefficients().zero());
        assert_eq!(poly(&[7]).evaluate(&int(9)), int(7));

        // `as_fn` gives a value usable where a function is expected.
        let p = poly(&[1, 2]); // 1 + 2x
        let f = p.as_fn();
        assert_eq!(
            (0..4).map(|n| f(&int(n))).collect::<Vec<_>>(),
            (0..4).map(|n| int(1 + 2 * n)).collect::<Vec<_>>()
        );

        // Reduction happens in the ring: evaluating over F_7 stays in F_7.
        let f7 = Integers::modulo(7);
        let f7x = f7.polynomials();
        let q = f7x.element(vec![f7.element(1), f7.element(0), f7.element(1)]); // 1 + x^2
        assert_eq!(q.evaluate(&f7.element(3)), f7.element(3)); // 1 + 9 = 10 ≡ 3
    }

    #[test]
    fn polynomial_division_in_f7_x() {
        let f7 = Integers::modulo(7);
        let f7x = f7.polynomials();
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
        let f7x = f7.polynomials();
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
        let f7x = f7.polynomials();
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

        // Compound assignment reads both operands by reference rather than cloning.
        let mut x = f7.element(3);
        x *= &f7.element(6);
        assert_eq!(x, f7.element(4));
        x += &f7.element(5);
        assert_eq!(x, f7.element(2));

        let mut n = int(10);
        n += &int(5);
        n *= &int(3);
        n -= &int(5);
        assert_eq!(n, int(40));

        // Same for polynomials, where the integer becomes a constant polynomial.
        let zx = Integers::default().polynomials();
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
        use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};

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
        impl AddAssign for Opaque {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
            }
        }
        impl MulAssign for Opaque {
            fn mul_assign(&mut self, rhs: Self) {
                self.0 *= rhs.0;
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
