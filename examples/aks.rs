//! The AKS deterministic primality test.
//!
//! Agrawal, Kayal and Saxena, "PRIMES is in P", Annals of Mathematics 160 (2004)
//! 781-793. Polynomial time, deterministic, and unconditional - but far slower in
//! practice than the probabilistic tests everyone actually uses.
//!
//! Run with `cargo run --release --example aks -- [n ...]`.

use num_bigint::BigInt;
use num_traits::{One, Zero};
use ruffini::integers::Integers;
use ruffini::polynomials::{Polynomial, RingExt};
use ruffini::structures::{Domain, EuclideanDomain, QuotientRing};
use std::rc::Rc;

type Zn = Rc<QuotientRing<Integers>>;
type ZnX = Rc<ruffini::polynomials::PolynomialRing<Zn>>;
type Cyclotomic = Rc<QuotientRing<ZnX>>;

fn main() {
    let args: Vec<BigInt> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let candidates = if args.is_empty() {
        vec![
            BigInt::from(31),
            BigInt::from(91),
            BigInt::from(991),
            BigInt::from(14197),
            BigInt::from(4294967297u64),
        ]
    } else {
        args
    };

    for n in candidates {
        let start = std::time::Instant::now();
        let prime = aks(&n);
        println!(
            "{n} is {}a prime  ({:?})",
            if prime { "" } else { "not " },
            start.elapsed()
        );
    }
}

fn aks(n: &BigInt) -> bool {
    if n < &BigInt::from(2) {
        return false;
    }
    let log2 = bit_length(n);
    let bound = log2 * log2;

    // Smallest r for which n has order above log2(n)^2 modulo r.
    let r = (2u64..)
        .find(|&r| order_exceeds(n, r, bound))
        .expect("some r has large enough order");

    // Trial division by everything up to r. This settles every small case, so the
    // polynomial stage only ever sees n with no small factors.
    for a in 2..=r {
        let a = BigInt::from(a);
        if n % &a == BigInt::zero() {
            return n == &a;
        }
    }
    if n <= &BigInt::from(r) {
        return true;
    }

    // Z_n[x] / (x^r - 1). Z_n is a field only when n is prime, but nothing here
    // inverts anything: x^r - 1 is monic, so reduction never needs a division.
    let zn = Integers::modulo(n.clone());
    let znx = zn.polynomials();
    let ring = znx.quotient(cyclotomic_modulus(&znx, r as usize));

    // x^n, computed once and reused for every witness. The exponent is `n` again,
    // owned, from the ring that already holds it as its modulus.
    let x = ring.element(znx.indeterminate());
    let x_to_n = x.pow(zn.order());

    // (x + a)^n == x^n + a must hold for every a below sqrt(phi(r)) * log2(n).
    let limit = (isqrt(totient(r)) * log2 as u64).max(1);
    (1..limit).all(|a| witness_agrees(&ring, &znx, &zn, &x_to_n, a))
}

/// Whether `(x + a)^n` and `x^n + a` agree in the ring.
fn witness_agrees(ring: &Cyclotomic, znx: &ZnX, zn: &Zn, x_to_n: &<Cyclotomic as Domain>::E, a: u64) -> bool {
    let shifted = ring.element(znx.indeterminate() + a as i64);
    shifted.pow(zn.order()) == x_to_n + a as i64
}

/// `x^r - 1` over `Z_n`.
fn cyclotomic_modulus(znx: &ZnX, r: usize) -> Polynomial<Zn> {
    let x = znx.indeterminate();
    x.pow(r) - 1
}

/// Whether the multiplicative order of `n` modulo `r` exceeds `bound`.
fn order_exceeds(n: &BigInt, r: u64, bound: u32) -> bool {
    let r = BigInt::from(r);
    if n % &r == BigInt::zero() {
        // n shares a factor with r, so it has no order at all.
        return false;
    }
    let mut power = BigInt::one();
    for _ in 0..=bound {
        power = power * n % &r;
        if power.is_one() {
            return false;
        }
    }
    true
}

/// Euler's totient, by trial division. `r` is small here, so this is fine.
fn totient(r: u64) -> u64 {
    let (mut remaining, mut result) = (r, r);
    let mut p = 2;
    while p * p <= remaining {
        if remaining % p == 0 {
            while remaining % p == 0 {
                remaining /= p;
            }
            result -= result / p;
        }
        p += 1;
    }
    if remaining > 1 {
        result -= result / remaining;
    }
    result
}

/// Number of bits in `n`, which is `ceil(log2(n))` rounded the way AKS wants.
fn bit_length(n: &BigInt) -> u32 {
    n.bits() as u32
}

fn isqrt(n: u64) -> u64 {
    if n == 0 {
        return 0;
    }
    let mut x = n;
    let mut y = x.div_ceil(2);
    while y < x {
        x = y;
        y = (x + n / x) / 2;
    }
    x
}
