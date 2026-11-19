//! The CGL hash, as a toy over `F_{p^2}`.
//!
//! Charles, Goren and Lauter's hash walks the supersingular 2-isogeny graph: each bit
//! of the input picks one of the two ways onward, and the j-invariant where the walk
//! stops is the digest. Two inputs collide exactly when their paths meet again, which
//! is to say when the graph has a cycle, and finding one is the hard problem.
//!
//! A step, from a curve `E` and the x-coordinate `x_b` it must not go back through:
//!
//! 1. The three points of order two are the roots of `x^3 + ax + b`, one of which is
//!    `x_b`. Dividing it out leaves `x^2 + x_b*x + x_b^2 + a`, and the quadratic
//!    formula gives the other two without searching the field.
//! 2. The two are ordered by a fixed order on `F_{p^2}`, here the coefficient of `i`
//!    and then the constant, both taken into `0..p`. Bit zero takes the smaller.
//!    Nothing here needs the order to be canonical, only to be the same every time.
//! 3. Vélu on that kernel gives the step, and the next curve.
//! 4. The unchosen root maps to the kernel of the dual, which is the root the next
//!    step must not take: going back through it would undo this one.
//!
//! This is the bare construction, so the length of the input is not encoded. That is
//! less of a gap than it looks: an input and an extension of it collide only when the
//! extension's tail is a cycle, which is what the whole thing already rests on. The
//! start curve is the real caveat, `y^2 = x^3 + x` having a known endomorphism ring,
//! which is what a parameter choice for use has to avoid.
//!
//! Charles, Goren and Lauter, "Cryptographic hash functions from expander graphs",
//! Journal of Cryptology 22 (2009) 93-113. Run with
//! `cargo run --release --bin cgl_hash -- [p]`, where `p` is a prime congruent to 3
//! modulo 4 and defaults to 83.

use num_bigint::BigInt;
use ruffini::elliptic::{Curve, Isogeny};
use ruffini::finite_fields::{finite_field, ExtensionField, FiniteField, PrimeField, Residue};
use ruffini::polynomials::PolynomialRing;
use ruffini::structures::{CommutativeMonoid, Field, QuotientRingElement, Ring};
use std::sync::Arc;

type Curves = Arc<Curve<ExtensionField>>;
type Element = QuotientRingElement<Arc<PolynomialRing<PrimeField>>>;

fn main() {
    let p: u32 = std::env::args()
        .nth(1)
        .map_or(83, |argument| argument.parse().expect("p must be a number"));
    assert_eq!(
        p % 4,
        3,
        "y^2 = x^3 + x is supersingular only for p = 3 (mod 4)"
    );
    println!("The CGL hash over F_{p}^2\n");

    // The start of every walk: y^2 = x^3 + x, with (0, 0) the point not to go back
    // through. That one is the loop at j = 1728, so excluding it costs no edges.
    let field = finite_field(p, 2);
    let start = Curve::new(field.clone(), field.from_integer(1), field.from_integer(0))
        .expect("nonsingular");
    let excluded = field.zero();

    println!("1. The start");
    println!(
        "   E_0: y^2 = x^3 + x over F_{p}^2, with j = {} and the kernel (0, 0) held back",
        show(p, &start.j_invariant())
    );

    println!("\n2. A walk");
    let bits = [true, false, true, true, false, false, true];
    let path = cgl_walk(&start, excluded.clone(), &bits);
    let mut j = show(p, &start.j_invariant());
    for (bit, step) in bits.iter().zip(&path) {
        let next = show(p, &step.codomain().j_invariant());
        println!("   bit {} takes j = {j} to j = {next}", u8::from(*bit));
        j = next;
    }
    println!(
        "   the digest of {} is j = {j}",
        bits.iter()
            .map(|b| u8::from(*b).to_string())
            .collect::<String>()
    );

    // Every input of ten bits, which is enough to reach the whole graph.
    println!("\n3. Where ten-bit inputs land");
    let digests = every_digest(&start, &excluded, 10);

    let mut vertices: Vec<(Element, usize)> = Vec::new();
    for (_, digest) in &digests {
        match vertices.iter_mut().find(|(j, _)| j == digest) {
            Some((_, count)) => *count += 1,
            None => vertices.push((digest.clone(), 1)),
        }
    }
    for (j, count) in &vertices {
        println!("   j = {:<12} {count} of 1024", show(p, j));
    }

    // Deuring's count again, as in the isogeny_graph demo.
    let correction = match p % 12 {
        1 => 0,
        5 | 7 => 1,
        _ => 2,
    };
    let expected = (p / 12 + correction) as usize;
    assert_eq!(
        vertices.len(),
        expected,
        "the walks reach {} of the {expected} vertices",
        vertices.len()
    );
    println!("   every one of the {expected} supersingular j-invariants is reached");

    println!("\n4. A collision");
    let (left, right) = digests
        .iter()
        .enumerate()
        .find_map(|(i, (bits, digest))| {
            digests[i + 1..]
                .iter()
                .find(|(_, other)| other == digest)
                .map(|(other, _)| (bits, other))
        })
        .expect("1024 inputs over a handful of vertices");
    println!(
        "   {} and {} both hash to j = {}",
        bitstring(left),
        bitstring(right),
        show(p, &cgl_hash(&start, excluded, left))
    );

    // Which is a cycle, read out: the two walks run together until they part, and the
    // two tails from there are a loop back to one vertex.
    let shared = left.iter().zip(right).take_while(|(a, b)| a == b).count();
    let cycle = left.len() + right.len() - 2 * shared;
    println!(
        "   they agree for {shared} steps, so this is a cycle of {cycle} steps at the\n\
         \x20  vertex they part at, which is what a cryptographic p is sized to hide"
    );
}

/// The isogenies the bits walk through, each from the codomain of the last.
///
/// `excluded` is the x-coordinate of the point of order two that the walk has just
/// come through, which it must not take again.
fn cgl_walk(start: &Curves, excluded: Element, bits: &[bool]) -> Vec<Isogeny<ExtensionField>> {
    let mut curve = Arc::clone(start);
    let mut excluded = excluded;
    let mut path = Vec::with_capacity(bits.len());

    for &bit in bits {
        let (step, next) = cgl_step(&curve, &excluded, bit);
        curve = Arc::clone(step.codomain());
        excluded = next;
        path.push(step);
    }
    path
}

/// One step: the isogeny the bit picks, and the root the step after it must not take.
fn cgl_step(curve: &Curves, excluded: &Element, bit: bool) -> (Isogeny<ExtensionField>, Element) {
    let (smaller, larger) = other_roots(curve, excluded);
    let (chosen, other) = if bit {
        (larger, smaller)
    } else {
        (smaller, larger)
    };

    let zero = curve.field().zero();
    let kernel = curve
        .point(chosen, zero.clone())
        .expect("a root of the cubic");
    let step = curve.isogeny(&kernel, 2);

    // The unchosen root maps to the generator of the dual's kernel, which is the way
    // back. The excluded root would map to the same point, the three points of order
    // two summing to infinity.
    let back = curve.point(other, zero).expect("a root of the cubic");
    let image = step.evaluate(&back);
    let (x, _) = image
        .coordinates()
        .expect("only the chosen root maps to infinity");
    let next = x.clone();
    (step, next)
}

/// The digests of every input of `length` bits, with the input that reached each.
///
/// The walks share their prefixes, so this is a tree rather than `2^length` walks:
/// `2^(length + 1)` steps in all rather than `length * 2^length`.
fn every_digest(start: &Curves, excluded: &Element, length: usize) -> Vec<(Vec<bool>, Element)> {
    let mut frontier = vec![(Arc::clone(start), excluded.clone(), Vec::new())];
    for _ in 0..length {
        let mut next = Vec::with_capacity(frontier.len() * 2);
        for (curve, excluded, bits) in &frontier {
            for bit in [false, true] {
                let (step, after) = cgl_step(curve, excluded, bit);
                let mut taken = bits.clone();
                taken.push(bit);
                next.push((Arc::clone(step.codomain()), after, taken));
            }
        }
        frontier = next;
    }
    frontier
        .into_iter()
        .map(|(curve, _, bits)| (bits, curve.j_invariant()))
        .collect()
}

/// The j-invariant the walk ends on, which is the digest.
fn cgl_hash(start: &Curves, excluded: Element, bits: &[bool]) -> Element {
    match cgl_walk(start, excluded, bits).last() {
        Some(step) => step.codomain().j_invariant(),
        None => start.j_invariant(),
    }
}

/// The two roots of the cubic other than `excluded`, smaller first.
///
/// `x^3 + ax + b = (x - r)(x^2 + rx + r^2 + a)`, so the other two are
/// `(-r +- sqrt(-3r^2 - 4a))/2`.
fn other_roots(curve: &Curves, excluded: &Element) -> (Element, Element) {
    let field = curve.field();
    let (a, _) = curve.coefficients();
    let discriminant = field.from_integer(-3) * excluded.clone() * excluded.clone()
        - field.from_integer(4) * a.clone();

    let root = field
        .sqrt(&discriminant)
        .expect("a supersingular curve over F_{p^2} has all three roots there");
    assert!(
        !field.is_zero(&root),
        "the other two points of order two coincide"
    );
    let half = field
        .invert(&field.from_integer(2))
        .expect("the characteristic is odd");

    let negated = field.zero() - excluded.clone();
    let first = (negated.clone() + root.clone()) * half.clone();
    let second = (negated - root) * half;
    let p = order_of_the_prime_field(field);
    if key(p, &first) <= key(p, &second) {
        (first, second)
    } else {
        (second, first)
    }
}

/// Where the order on `F_{p^2}` comes from: the coefficient of `i`, then the constant,
/// both taken into `0..p`. Any order the walk can repeat would do.
fn key(p: u32, x: &Element) -> (BigInt, BigInt) {
    let coefficient = |i: usize| canonical(p, &x.representative().coefficient(i));
    (coefficient(1), coefficient(0))
}

/// A coefficient as an integer in `0..p`, since the library reduces representatives
/// without canonicalising them.
fn canonical(p: u32, residue: &Residue) -> BigInt {
    let modulus = BigInt::from(p);
    let value: BigInt = residue.representative().clone().into();
    ((value % &modulus) + &modulus) % &modulus
}

/// `c1*i + c0`, with both coefficients in `0..p`.
fn show(p: u32, x: &Element) -> String {
    let (linear, constant) = key(p, x);
    match (linear.to_string().as_str(), constant.to_string().as_str()) {
        ("0", c) => c.to_string(),
        ("1", "0") => "i".to_string(),
        (l, "0") => format!("{l}i"),
        ("1", c) => format!("i + {c}"),
        (l, c) => format!("{l}i + {c}"),
    }
}

fn bitstring(bits: &[bool]) -> String {
    bits.iter().map(|b| u8::from(*b).to_string()).collect()
}

/// The characteristic, which the key needs to reduce coefficients.
fn order_of_the_prime_field(field: &ExtensionField) -> u32 {
    field
        .characteristic()
        .try_into()
        .expect("a small characteristic")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `F_{p^2}`, the start curve and the root it holds back.
    fn setup(p: u32) -> (ExtensionField, Curves, Element) {
        let field = finite_field(p, 2);
        let start = Curve::new(field.clone(), field.from_integer(1), field.from_integer(0))
            .expect("nonsingular");
        let excluded = field.zero();
        (field, start, excluded)
    }

    /// The x-coordinates of the points of order two, by trying every element of the
    /// field: what the walk works out from the excluded root, found without it.
    fn two_torsion(curve: &Curves) -> Vec<Element> {
        let field = curve.field();
        let order = field.order();
        let mut roots = Vec::new();
        let mut index = BigInt::from(0u32);
        while index < order {
            let x = field.element_at(&index);
            if field.is_zero(&curve.evaluate(&x)) {
                roots.push(x);
            }
            index += 1u32;
        }
        roots
    }

    const BITS: [bool; 8] = [true, false, false, true, true, false, true, false];

    #[test]
    fn the_same_input_hashes_the_same_way() {
        let (_, start, excluded) = setup(23);
        let digest = cgl_hash(&start, excluded.clone(), &BITS);
        assert_eq!(cgl_hash(&start, excluded, &BITS), digest);
    }

    #[test]
    fn the_walk_stays_in_the_supersingular_graph() {
        let (_, start, excluded) = setup(23);
        assert!(start.is_supersingular());
        for step in cgl_walk(&start, excluded, &BITS) {
            assert!(step.codomain().is_supersingular());
        }
    }

    #[test]
    fn the_walk_never_turns_back() {
        // A step back would make the composite of the two isomorphic to [2], which
        // kills all of E[2]. So some point of order two has to survive it, and the
        // 2-torsion is found here without the exclusion the walk uses.
        let (field, start, excluded) = setup(23);
        let path = cgl_walk(&start, excluded, &BITS);
        for pair in path.windows(2) {
            let (phi, psi) = (&pair[0], &pair[1]);
            let domain = phi.domain();
            let survives = two_torsion(domain).into_iter().any(|x| {
                let t = domain.point(x, field.zero()).expect("a root of the cubic");
                !psi.evaluate(&phi.evaluate(&t)).is_infinity()
            });
            assert!(
                survives,
                "the composite killed all of E[2]: the walk went back"
            );
        }
    }

    #[test]
    fn every_step_is_an_edge_of_the_graph() {
        // The adjacency the isogeny_graph demo prints, recomputed from the three
        // roots of each curve's cubic rather than from the step the walk took.
        let (field, start, excluded) = setup(23);
        for step in cgl_walk(&start, excluded, &BITS) {
            let domain = step.domain();
            let neighbours: Vec<Element> = two_torsion(domain)
                .into_iter()
                .map(|x| {
                    let t = domain.point(x, field.zero()).expect("a root of the cubic");
                    domain.isogeny(&t, 2).codomain().j_invariant()
                })
                .collect();
            assert!(
                neighbours.contains(&step.codomain().j_invariant()),
                "the step went somewhere no 2-isogeny goes"
            );
        }
    }

    #[test]
    fn the_digests_reach_every_vertex() {
        // F_23^2 has three supersingular j-invariants, and eight-bit inputs find all
        // of them. The p = 83 sweep over all 1024 ten-bit inputs is the demo itself,
        // which is slow enough to want a release build.
        let (_, start, excluded) = setup(23);
        let digests = every_digest(&start, &excluded, 8);
        assert_eq!(digests.len(), 256);

        let mut vertices: Vec<Element> = Vec::new();
        for (_, digest) in &digests {
            if !vertices.contains(digest) {
                vertices.push(digest.clone());
            }
        }
        assert_eq!(vertices.len(), 23 / 12 + 2, "floor(p/12) + 2 vertices");
    }
}
