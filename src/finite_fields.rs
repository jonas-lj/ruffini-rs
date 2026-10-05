//! Finite fields `F_{p^k}`, as `F_p[x]` modulo an irreducible polynomial of degree `k`.
//!
//! The prime case needs none of this: `F_p` is [`Integers::modulo`] with a prime
//! modulus. What an extension needs is an irreducible polynomial to quotient by, which
//! this module either looks up or finds.

use crate::euclidean::extended_gcd;
use crate::integers::Integers;
use crate::number_theory::factorise;
use crate::polynomials::{Polynomial, PolynomialRing, RingExt};
use crate::structures::{Domain, EuclideanDomain, Monoid, QuotientRing};
use num_bigint::BigInt;
use std::sync::Arc;

/// `F_p`, the integers modulo a prime.
pub type PrimeField = Arc<QuotientRing<Integers>>;

/// `F_{p^k}`, a quotient of `F_p[x]` by an irreducible polynomial of degree `k`.
pub type ExtensionField = Arc<QuotientRing<Arc<PolynomialRing<PrimeField>>>>;

/// Irreducible polynomials over `F_p`, as coefficients in increasing degree, for the
/// `(p, k)` pairs listed. Each is the lexicographically least that [`is_irreducible`]
/// accepts, and the tests check every entry against it.
///
/// The table only fixes the common cases so that the same `(p, k)` always gives the
/// same field. [`irreducible`] searches when a pair is not listed, which is cheap:
/// about one monic polynomial in `k` is irreducible, so a handful of candidates suffice.
const TABLE: &[(u32, &[&[i64]])] = &[
    (
        2,
        &[
            &[0, 1],
            &[1, 1, 1],
            &[1, 1, 0, 1],
            &[1, 1, 0, 0, 1],
            &[1, 0, 1, 0, 0, 1],
            &[1, 1, 0, 0, 0, 0, 1],
            &[1, 1, 0, 0, 0, 0, 0, 1],
            &[1, 1, 0, 1, 1, 0, 0, 0, 1],
        ],
    ),
    (
        3,
        &[
            &[0, 1],
            &[1, 0, 1],
            &[1, 2, 0, 1],
            &[2, 1, 0, 0, 1],
            &[1, 2, 0, 0, 0, 1],
            &[2, 1, 0, 0, 0, 0, 1],
            &[2, 0, 1, 0, 0, 0, 0, 1],
            &[2, 0, 1, 0, 0, 0, 0, 0, 1],
        ],
    ),
    (
        5,
        &[
            &[0, 1],
            &[2, 0, 1],
            &[1, 1, 0, 1],
            &[2, 0, 0, 0, 1],
            &[1, 4, 0, 0, 0, 1],
            &[2, 1, 0, 0, 0, 0, 1],
            &[1, 1, 0, 0, 0, 0, 0, 1],
            &[2, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
    ),
    (
        7,
        &[
            &[0, 1],
            &[1, 0, 1],
            &[2, 0, 0, 1],
            &[1, 1, 0, 0, 1],
            &[3, 1, 0, 0, 0, 1],
            &[2, 0, 0, 0, 0, 0, 1],
            &[1, 6, 0, 0, 0, 0, 0, 1],
            &[3, 1, 0, 0, 0, 0, 0, 0, 1],
        ],
    ),
];

/// An irreducible monic polynomial of degree `k` over `F_p`.
///
/// # Panics
/// If `k` is zero, or if `p` is not the modulus of a prime field.
pub fn irreducible(ring: &Arc<PolynomialRing<PrimeField>>, k: usize) -> Polynomial<PrimeField> {
    assert!(
        k > 0,
        "degree of an irreducible polynomial must be positive"
    );
    let field = ring.coefficients();
    match tabulated(&field.order(), k) {
        Some(coefficients) => ring.element(
            coefficients
                .iter()
                .map(|c| field.element(*c))
                .collect::<Vec<_>>(),
        ),
        None => search(ring, k),
    }
}

/// The lexicographically least monic irreducible of degree `k`, by the lower
/// coefficients, found by testing candidates in that order.
fn search(ring: &Arc<PolynomialRing<PrimeField>>, k: usize) -> Polynomial<PrimeField> {
    let field = ring.coefficients();
    let p = field.order();
    let mut lower = vec![BigInt::from(0); k];
    loop {
        let mut coefficients: Vec<_> = lower.iter().map(|c| field.element(c.clone())).collect();
        coefficients.push(field.identity());
        let candidate = ring.element(coefficients);
        if is_irreducible(ring, &candidate) {
            return candidate;
        }

        let mut i = 0;
        loop {
            assert!(i < k, "no irreducible polynomial of degree {k} found");
            lower[i] += 1;
            if lower[i] < p {
                break;
            }
            lower[i] = BigInt::from(0);
            i += 1;
        }
    }
}

/// `F_{p^k}`, for a prime `p`.
///
/// `k == 1` still goes through a polynomial quotient, so that the type does not depend
/// on the exponent; use [`Integers::modulo`] directly for a prime field.
///
/// Primality of `p` is the caller's to guarantee, as it is for [`Integers::modulo`].
///
/// # Panics
/// If `k` is zero.
pub fn finite_field(p: impl Into<BigInt>, k: usize) -> ExtensionField {
    // One ring handle for both the modulus and the quotient: elements built from two
    // separate `polynomials()` calls belong to different rings.
    let ring = Integers::modulo(p.into()).polynomials();
    let modulus = irreducible(&ring, k);
    ring.quotient(modulus)
}

/// Whether `f` is irreducible over `F_p`, by Rabin's test.
///
/// `f` of degree `k` is irreducible exactly when `x^(p^k) = x` in `F_p[x]/(f)` while
/// `gcd(x^(p^(k/q)) - x, f) = 1` for every prime `q` dividing `k`: the first says every
/// root lies in `F_{p^k}`, the others that none lies in a proper subfield.
///
/// # Panics
/// If `f` is zero.
pub fn is_irreducible(ring: &Arc<PolynomialRing<PrimeField>>, f: &Polynomial<PrimeField>) -> bool {
    let k = f.degree().expect("the zero polynomial is not irreducible");
    if k == 0 {
        return false;
    }
    if k == 1 {
        return true;
    }

    let field = ring.coefficients().clone();
    let p = field.order();
    let quotient = ring.quotient(f.clone());
    let x = quotient.element(ring.indeterminate());

    for (q, _) in factorise(k as u64) {
        let exponent = q
            .try_into()
            .map(|q: u32| p.pow(k as u32 / q))
            .expect("a prime divisor of the degree fits a machine integer");
        let power = x.pow(exponent);
        // The gcd needs the Euclidean domain, not its quotient. A zero difference means
        // x already lies in the subfield, and gcd(0, f) is f, so it fails the test too.
        let difference = power.representative().clone() - ring.indeterminate();
        let (gcd, _, _) = extended_gcd(&ring.clone(), difference, f.clone());
        if gcd != ring.identity() {
            return false;
        }
    }

    x.pow(p.pow(k as u32)) == x
}

fn tabulated(p: &BigInt, k: usize) -> Option<&'static [i64]> {
    let p: u32 = p.try_into().ok()?;
    let entries = TABLE.iter().find(|(prime, _)| *prime == p)?.1;
    entries.get(k - 1).copied()
}

#[cfg(test)]
mod generate {
    use super::*;

    /// Prints the table. Not a check - run it when the table needs regenerating.
    #[test]
    #[ignore]
    fn table() {
        for p in [2u32, 3, 5, 7] {
            let field = Integers::modulo(p);
            let rows: Vec<String> = (1..=8)
                .map(|k| {
                    let f = search(&field.polynomials(), k);
                    let cs: Vec<String> = f
                        .coefficients()
                        .iter()
                        .map(|c| format!("{}", c.representative()))
                        .collect();
                    format!("            &[{}],", cs.join(", "))
                })
                .collect();
            println!(
                "    (\n        {p},\n        &[\n{}\n        ],\n    ),",
                rows.join("\n")
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{CommutativeMonoid, DivRem, Field};

    /// Every monic polynomial of the given degree over `F_p`.
    fn monic(ring: &Arc<PolynomialRing<PrimeField>>, degree: usize) -> Vec<Polynomial<PrimeField>> {
        let field = ring.coefficients();
        let p: u32 = field.order().try_into().unwrap();
        let count = p.pow(degree as u32);
        (0..count)
            .map(|n| {
                let mut coefficients: Vec<_> = (0..degree)
                    .map(|i| field.element((n / p.pow(i as u32) % p) as i64))
                    .collect();
                coefficients.push(field.identity());
                ring.element(coefficients)
            })
            .collect()
    }

    /// Irreducibility by trial division, which is the definition rather than a test for
    /// it: over a field any factor can be made monic, so monic divisors are enough.
    fn divides_nothing_smaller(
        ring: &Arc<PolynomialRing<PrimeField>>,
        f: &Polynomial<PrimeField>,
    ) -> bool {
        let k = f.degree().unwrap();
        if k < 2 {
            return k == 1;
        }
        (1..=k / 2).all(|d| monic(ring, d).iter().all(|g| f.div_rem(g).1 != ring.zero()))
    }

    #[test]
    fn rabins_test_agrees_with_trial_division() {
        for (p, max_degree) in [(2u32, 5usize), (3, 4), (5, 3), (7, 3)] {
            let field = Integers::modulo(p);
            let ring = field.polynomials();
            for k in 1..=max_degree {
                for f in monic(&ring, k) {
                    assert_eq!(
                        is_irreducible(&ring, &f),
                        divides_nothing_smaller(&ring, &f),
                        "F_{p}, degree {k}: {f}"
                    );
                }
            }
        }
    }

    #[test]
    fn known_irreducibles_and_reducibles() {
        let f2 = Integers::modulo(2);
        let r2 = f2.polynomials();
        let x = r2.indeterminate();
        assert!(is_irreducible(&r2, &(x.pow(2) + x.clone() + 1)));
        assert!(is_irreducible(&r2, &(x.pow(3) + x.clone() + 1)));
        // x^2 + x is x(x + 1), and over F_2 x^2 + 1 is (x + 1)^2.
        assert!(!is_irreducible(&r2, &(x.pow(2) + x.clone())));
        assert!(!is_irreducible(&r2, &(x.pow(2) + 1)));
        assert!(!is_irreducible(&r2, &(x.pow(4) + 1)));
        // Degree one is always irreducible, degree zero never.
        assert!(is_irreducible(&r2, &x));
        assert!(!is_irreducible(&r2, &r2.identity()));

        let f3 = Integers::modulo(3);
        let r3 = f3.polynomials();
        let y = r3.indeterminate();
        // -1 is not a square mod 3, so y^2 + 1 has no root; mod 7, 3^2 = 2.
        assert!(is_irreducible(&r3, &(y.pow(2) + 1)));
        let f7 = Integers::modulo(7);
        let r7 = f7.polynomials();
        let z = r7.indeterminate();
        assert!(!is_irreducible(&r7, &(z.pow(2) - 2)));
    }

    #[test]
    fn the_table_matches_the_search_and_is_irreducible() {
        for (p, entries) in TABLE {
            let field = Integers::modulo(*p);
            let ring = field.polynomials();
            for (i, _) in entries.iter().enumerate() {
                let k = i + 1;
                let tabulated = irreducible(&ring, k);
                assert_eq!(tabulated.degree(), Some(k), "F_{p}, degree {k}");
                assert!(is_irreducible(&ring, &tabulated), "F_{p}, degree {k}");
                assert_eq!(tabulated, search(&ring, k), "F_{p}, degree {k}");
            }
        }
    }

    #[test]
    fn extension_fields_have_the_right_order_and_invert() {
        for (p, k) in [(2u32, 2usize), (2, 3), (3, 2), (5, 2), (2, 8)] {
            let order = BigInt::from(p).pow(k as u32);
            let field = finite_field(p, k);
            // The field's own polynomial ring, not a fresh one: `base()` is how a
            // caller reaches it to build elements at all.
            let ring = field.base().clone();
            let base = ring.coefficients().clone();

            // Every element is a polynomial of degree below k, and there are p^k of
            // them. Fermat's little theorem in the extension: a^(p^k - 1) = 1.
            let mut rows: Vec<Vec<_>> = vec![vec![]];
            for _ in 0..k {
                rows = rows
                    .iter()
                    .flat_map(|prefix| {
                        (0..p).map(|c| {
                            let mut next = prefix.clone();
                            next.push(base.element(c as i64));
                            next
                        })
                    })
                    .collect();
            }
            let elements: Vec<_> = rows
                .into_iter()
                .map(|cs| field.element(ring.element(cs)))
                .collect();
            assert_eq!(elements.len(), order.clone().try_into().unwrap());

            for a in &elements {
                if field.is_zero(a) {
                    assert_eq!(field.invert(a), None);
                    continue;
                }
                assert_eq!(a.pow(&order - 1), field.identity(), "F_{p}^{k}");
                let inverse = field.invert(a).expect("a nonzero element inverts");
                assert_eq!(a * &inverse, field.identity(), "F_{p}^{k}");
            }
        }
    }

    #[test]
    fn the_characteristic_is_p() {
        // Adding 1 to itself p times gives zero, which is what distinguishes F_4 from
        // Z/4: here 2 = 0 rather than 2 being a zero divisor.
        for (p, k) in [(2u32, 2usize), (3, 2), (5, 3)] {
            let field = finite_field(p, k);
            let one = field.identity();
            let sum = (0..p).fold(field.zero(), |acc, _| acc + one.clone());
            assert!(field.is_zero(&sum), "F_{p}^{k}");
        }
    }
}
