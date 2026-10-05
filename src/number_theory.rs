//! Elementary number theory over [`BigInt`]: factorisation, Euler's totient, and
//! multiplicative order.
//!
//! Trial division throughout, so these suit the small moduli that show up as
//! parameters - the `r` of a primality test, the order of a small group - rather than
//! the large ones a factoring attack would aim at.

use num_bigint::BigInt;
use num_traits::{One, Signed, Zero};

/// The prime factorisation of `n`, each prime with its multiplicity, in increasing
/// order. Returns nothing for `0`, `1` and `-1`; the sign is otherwise ignored.
pub fn factorise(n: impl Into<BigInt>) -> Vec<(BigInt, u32)> {
    let mut remaining = n.into().abs();
    let mut factors = Vec::new();
    let mut p = BigInt::from(2);
    while &p * &p <= remaining {
        let mut multiplicity = 0;
        while (&remaining % &p).is_zero() {
            remaining /= &p;
            multiplicity += 1;
        }
        if multiplicity > 0 {
            factors.push((p.clone(), multiplicity));
        }
        p += 1;
    }
    if remaining > BigInt::one() {
        factors.push((remaining, 1));
    }
    factors
}

/// Euler's totient: how many of `1..n` are coprime to `n`.
///
/// # Panics
/// If `n` is not positive.
pub fn totient(n: impl Into<BigInt>) -> BigInt {
    let n = n.into();
    assert!(n.is_positive(), "totient of a non-positive number");
    factorise(n.clone())
        .into_iter()
        .fold(n, |acc, (p, _)| acc / &p * (p - 1))
}

/// The multiplicative order of `a` modulo `n`: the least `k > 0` with `a^k = 1`.
///
/// [`None`] when `a` and `n` share a factor, since `a` then generates no cyclic
/// subgroup of the units.
///
/// The order divides the totient, so this divides prime factors out of the totient for
/// as long as the power still comes back to one, rather than multiplying up from `1`.
///
/// # Panics
/// If `n` is not positive.
pub fn multiplicative_order(a: impl Into<BigInt>, n: impl Into<BigInt>) -> Option<BigInt> {
    let (a, n) = (a.into(), n.into());
    assert!(
        n.is_positive(),
        "multiplicative order to a non-positive modulus"
    );
    let a = a.modpow(&BigInt::one(), &n);
    if gcd(a.clone(), n.clone()) != BigInt::one() {
        return None;
    }

    let mut order = totient(n.clone());
    for (p, _) in factorise(order.clone()) {
        while (&order % &p).is_zero() && a.modpow(&(&order / &p), &n).is_one() {
            order /= &p;
        }
    }
    Some(order)
}

/// The non-negative greatest common divisor, by the Euclidean algorithm.
fn gcd(mut a: BigInt, mut b: BigInt) -> BigInt {
    while !b.is_zero() {
        let r = a % &b;
        a = std::mem::replace(&mut b, r);
    }
    a.abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn big(n: i64) -> BigInt {
        BigInt::from(n)
    }

    #[test]
    fn factorisation_multiplies_back_to_the_input() {
        assert_eq!(factorise(1), vec![]);
        assert_eq!(factorise(2), vec![(big(2), 1)]);
        assert_eq!(factorise(360), vec![(big(2), 3), (big(3), 2), (big(5), 1)]);
        // A prime, and a prime squared, which the loop bound has to let through.
        assert_eq!(factorise(97), vec![(big(97), 1)]);
        assert_eq!(factorise(9409), vec![(big(97), 2)]);

        for n in 1..200i64 {
            let product = factorise(n)
                .into_iter()
                .fold(BigInt::one(), |acc, (p, k)| acc * p.pow(k));
            assert_eq!(product, big(n), "at n = {n}");
        }
    }

    #[test]
    fn totient_counts_the_coprime_residues() {
        // Against the definition, which is what the factorisation formula replaces.
        for n in 1..200u32 {
            let counted = (1..=n)
                .filter(|a| gcd(big(*a as i64), big(n as i64)).is_one())
                .count();
            assert_eq!(totient(n), big(counted as i64), "at n = {n}");
        }

        assert_eq!(totient(1), big(1));
        assert_eq!(totient(97), big(96));
        assert_eq!(totient(360), big(96));
    }

    #[test]
    #[should_panic(expected = "totient of a non-positive")]
    fn totient_rejects_zero() {
        totient(0);
    }

    #[test]
    fn multiplicative_order_is_the_least_returning_power() {
        // 2 has order 8 mod 17, and 3 is a primitive root mod 7.
        assert_eq!(multiplicative_order(2, 17), Some(big(8)));
        assert_eq!(multiplicative_order(3, 7), Some(big(6)));
        assert_eq!(multiplicative_order(1, 7), Some(big(1)));

        // Sharing a factor leaves no order at all.
        assert_eq!(multiplicative_order(6, 9), None);
        assert_eq!(multiplicative_order(0, 5), None);

        // Negative and oversized inputs reduce first.
        assert_eq!(multiplicative_order(-1, 7), Some(big(2)));
        assert_eq!(multiplicative_order(19, 17), multiplicative_order(2, 17));

        // Against a search for the least returning power.
        for n in 2..60i64 {
            for a in 0..n {
                let least = (1..=200)
                    .find(|k| big(a).modpow(&big(*k), &big(n)).is_one())
                    .map(big);
                assert_eq!(multiplicative_order(a, n), least, "at a = {a}, n = {n}");
            }
        }
    }
}
