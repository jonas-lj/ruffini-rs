//! The AKS deterministic primality test.
//!
//! Agrawal, Kayal and Saxena, "PRIMES is in P", Annals of Mathematics 160 (2004)
//! 781-793. Polynomial time, deterministic, and unconditional - but far slower in
//! practice than the probabilistic tests everyone actually uses.
//!
//! Run with `cargo run --release --example aks -- [n ...]`.

use num_bigint::BigInt;
use num_traits::{ToPrimitive, Zero};
use ruffini::integers::Integers;
use ruffini::number_theory::{multiplicative_order, totient};
use ruffini::polynomials::{Polynomial, RingExt};
use ruffini::structures::{Domain, EuclideanDomain, QuotientRing};
use std::sync::Arc;

type Zn = Arc<QuotientRing<Integers>>;
type ZnX = Arc<ruffini::polynomials::PolynomialRing<Zn>>;
type Cyclotomic = Arc<QuotientRing<ZnX>>;
type Residue = <Cyclotomic as Domain>::E;

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
    let bound = BigInt::from(log2 * log2);

    // Smallest r for which n has order above log2(n)^2 modulo r.
    let r = (2u64..)
        .find(|&r| multiplicative_order(n.clone(), r).is_some_and(|order| order > bound))
        .expect("some r has large enough order");

    // Trial division by everything up to r. This settles every small case, so the
    // polynomial stage only ever sees n with no small factors.
    for a in 2..=r {
        let a: BigInt = a.into();
        if (n % &a).is_zero() {
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

    // x^n, computed once and reused for every witness.
    let x = ring.element(znx.indeterminate());
    let x_to_n = x.pow(n.clone());

    // (x + a)^n == x^n + a must hold for every a below sqrt(phi(r)) * log2(n).
    let phi = totient(r).to_u64().expect("r is a machine integer");
    let limit = (phi.isqrt() * log2 as u64).max(1);
    (1..limit).all(|a| witness_agrees(n, &x, &x_to_n, a as i64))
}

/// Whether `(x + a)^n` and `x^n + a` agree in the ring.
fn witness_agrees(n: &BigInt, x: &Residue, x_to_n: &Residue, a: i64) -> bool {
    (x + a).pow(n.clone()) == x_to_n + a
}

/// `x^r - 1` over `Z_n`.
fn cyclotomic_modulus(znx: &ZnX, r: usize) -> Polynomial<Zn> {
    let x = znx.indeterminate();
    x.pow(r) - 1
}

/// Number of bits in `n`, which is `ceil(log2(n))` rounded the way AKS wants.
fn bit_length(n: &BigInt) -> u32 {
    n.bits() as u32
}
