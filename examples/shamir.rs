//! Shamir's secret sharing over a prime field.
//!
//! Shamir, "How to share a secret", Communications of the ACM 22 (1979) 612-613. The
//! secret is the constant term of a polynomial of degree `t - 1` whose other
//! coefficients are random, and the shares are its values at 1, 2, ..., n. Any `t` of
//! them determine the polynomial by interpolation, and so the secret; any `t - 1` leave
//! every secret equally possible, which this demonstrates rather than asserts.
//!
//! Not a secure implementation. The coefficients come from a fixed sequence so that the
//! output is reproducible, where real use needs them uniform and unguessable.
//!
//! Run with `cargo run --release --example shamir -- [secret] [threshold] [shares]`.

use itertools::Itertools;
use num_bigint::BigInt;
use ruffini::integers::Integers;
use ruffini::interpolation::interpolate;
use ruffini::polynomials::{PolynomialRing, RingExt};
use ruffini::structures::{CommutativeMonoid, Domain, Monoid, QuotientRing, QuotientRingElement};
use std::sync::Arc;

type Field = Arc<QuotientRing<Integers>>;
type Element = QuotientRingElement<Integers>;
type Polynomials = Arc<PolynomialRing<Field>>;

/// The Mersenne prime 2^61 - 1, big enough to hold a secret of sixty bits.
const PRIME: u64 = 2_305_843_009_213_693_951;

fn main() {
    let mut args = std::env::args().skip(1);
    let secret: u64 = parse(args.next(), 1_234_567_890_123);
    let threshold: usize = parse(args.next(), 3);
    let shares: usize = parse(args.next(), 5);
    assert!(threshold >= 1 && shares >= threshold, "need n >= t >= 1");

    let field = Integers::modulo(PRIME);
    let ring = field.polynomials();

    // f(x) = secret + a_1 x + ... + a_{t-1} x^{t-1}.
    let mut coefficients = vec![field.element(secret)];
    coefficients.extend((0..threshold - 1).map(|i| field.element(pseudorandom(i as u64))));
    let f = ring.element(coefficients);

    // The shares are f at 1, 2, ..., n - an arithmetic progression, so one pass of
    // forward differences gives them all.
    let one = field.identity();
    let dealt: Vec<_> = f.evaluations(&one, &one).take(shares).collect();

    println!("secret    {secret}");
    println!("threshold {threshold} of {shares}");
    for (i, share) in dealt.iter().enumerate() {
        println!("  share {:2}  {share}", i + 1);
    }

    // Every subset of t shares recovers the same secret.
    let mut recovered = 0;
    for subset in (1..=shares).combinations(threshold) {
        let xs: Vec<Element> = subset.iter().map(|i| field.element(*i as u64)).collect();
        let ys: Vec<Element> = subset.iter().map(|i| dealt[i - 1].clone()).collect();
        let p = interpolate(&ring, &xs, &ys).expect("the x values are distinct");
        assert_eq!(p, f, "subset {subset:?} rebuilt a different polynomial");
        assert_eq!(p.evaluate(&field.zero()), field.element(secret));
        recovered += 1;
    }
    println!("recovered the secret from all {recovered} subsets of {threshold} shares");

    if threshold >= 2 {
        demonstrate_that_one_share_short_tells_nothing(&field, &ring, &dealt, threshold, secret);
    }
}

/// With `t - 1` shares every secret is still possible: for any value at all there is a
/// polynomial of degree `t - 1` through those shares taking it at zero.
fn demonstrate_that_one_share_short_tells_nothing(
    field: &Field,
    ring: &Polynomials,
    dealt: &[Element],
    threshold: usize,
    secret: u64,
) {
    let short: Vec<usize> = (1..threshold).collect();
    let xs: Vec<Element> = short.iter().map(|i| field.element(*i as u64)).collect();
    let ys: Vec<Element> = short.iter().map(|i| dealt[i - 1].clone()).collect();

    println!("with only {} shares:", short.len());
    for candidate in [secret, secret + 1, 0, PRIME - 1] {
        // Pin the constant term to the candidate and interpolate through the shares.
        let mut xs = xs.clone();
        let mut ys = ys.clone();
        xs.push(field.zero());
        ys.push(field.element(candidate));
        let p = interpolate(ring, &xs, &ys).expect("the x values are distinct");

        let consistent = short
            .iter()
            .all(|i| p.evaluate(&field.element(*i as u64)) == dealt[i - 1]);
        println!("  a secret of {candidate:>19} is consistent with every share: {consistent}");
        assert!(consistent);
    }
}

fn parse<T: std::str::FromStr>(arg: Option<String>, default: T) -> T {
    arg.and_then(|a| a.parse().ok()).unwrap_or(default)
}

/// A fixed sequence standing in for randomness, which is what makes this a
/// demonstration of the algebra and not a secret-sharing implementation.
fn pseudorandom(i: u64) -> BigInt {
    let mut state = i.wrapping_add(1).wrapping_mul(6364136223846793005);
    state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    BigInt::from(state % PRIME)
}
