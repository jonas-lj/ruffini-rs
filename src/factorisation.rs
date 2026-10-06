//! Factorisation of polynomials over a prime field, by Cantor and Zassenhaus.
//!
//! Three stages, each narrowing what is left to do: squarefree decomposition splits off
//! the multiplicities, distinct-degree factorisation groups the remaining factors by
//! their degree, and equal-degree splitting separates the factors within a group.

use crate::euclidean::extended_gcd;
use crate::finite_fields::PrimeField;
use crate::polynomials::{Polynomial, PolynomialRing};
use crate::structures::{
    CommutativeMonoid, DivRem, Domain, EuclideanDomain, Field, Monoid, QuotientRing,
};
use num_bigint::BigInt;
use std::sync::Arc;

type Ring = Arc<PolynomialRing<PrimeField>>;
type Poly = Polynomial<PrimeField>;

/// The monic irreducible factors of `f`, each with its multiplicity, by increasing
/// degree. `f` equals its leading coefficient times the product of these.
///
/// # Panics
/// If `f` is zero, which has no factorisation.
pub fn factorise(ring: &Ring, f: &Poly) -> Vec<(Poly, u32)> {
    let mut factors = Vec::new();
    for (part, multiplicity) in squarefree(ring, f) {
        for (block, degree) in distinct_degree(ring, &part) {
            for factor in equal_degree(ring, &block, degree) {
                factors.push((factor, multiplicity));
            }
        }
    }
    factors.sort_by_key(|(q, _)| sort_key(ring, q));
    factors
}

/// Squarefree decomposition: pairwise coprime squarefree polynomials, each paired with
/// the multiplicity it has in `f`.
///
/// Musser's algorithm, which unlike Yun's survives characteristic `p` dividing a
/// multiplicity - the case where the derivative loses the factor entirely.
///
/// # Panics
/// If `f` is zero.
pub fn squarefree(ring: &Ring, f: &Poly) -> Vec<(Poly, u32)> {
    let one = ring.identity();
    let f = monic(ring, f);
    if f == one {
        return Vec::new();
    }

    let derivative = f.derivative();
    if derivative == ring.zero() {
        // f is a polynomial in x^p, so f = g^p and every multiplicity scales by p.
        return raise(ring, &pth_root(ring, &f));
    }

    let mut parts = Vec::new();
    let mut repeated = gcd(ring, f.clone(), derivative);
    let mut distinct = f.div_rem(&repeated).0;
    let mut multiplicity = 1;
    while distinct != one {
        let shared = gcd(ring, distinct.clone(), repeated.clone());
        let exact = distinct.div_rem(&shared).0;
        if exact != one {
            parts.push((exact, multiplicity));
        }
        distinct = shared.clone();
        repeated = repeated.div_rem(&shared).0;
        multiplicity += 1;
    }
    // Whatever the derivative could not see is a p-th power.
    if repeated != one {
        parts.extend(raise(ring, &pth_root(ring, &repeated)));
    }
    parts
}

/// Groups the factors of a squarefree `f` by degree: each entry is the product of all
/// the irreducible factors of `f` of that degree.
///
/// `x^(p^d) - x` is divisible by exactly the irreducibles whose degree divides `d`, so
/// taking its gcd with what is left at each `d` peels off one degree at a time.
///
/// # Panics
/// If `f` is zero.
pub fn distinct_degree(ring: &Ring, f: &Poly) -> Vec<(Poly, usize)> {
    let one = ring.identity();
    let mut blocks = Vec::new();
    let mut remaining = monic(ring, f);
    let mut degree = 1;

    while remaining.degree().unwrap_or(0) >= 2 * degree {
        let power = frobenius(ring, &remaining, degree);
        let block = gcd(ring, remaining.clone(), power - ring.indeterminate());
        if block != one {
            remaining = remaining.div_rem(&block).0;
            blocks.push((block, degree));
        }
        degree += 1;
    }
    if remaining != one {
        let degree = remaining.degree().expect("a non-unit is nonzero");
        blocks.push((remaining, degree));
    }
    blocks
}

/// Splits a product of distinct irreducibles that all have degree `degree`.
///
/// Cantor and Zassenhaus: for odd `p`, a random `a` has `a^((p^d - 1)/2)` equal to one
/// modulo some factors and not others, so the gcd with `a^((p^d - 1)/2) - 1` separates
/// them. In characteristic two there are no square roots of unity to exploit and the
/// trace map `a + a^2 + ... + a^(2^(d-1))` takes its place.
///
/// Candidates are tried in a fixed order rather than drawn at random, which keeps the
/// result reproducible and the crate free of a generator.
fn equal_degree(ring: &Ring, f: &Poly, degree: usize) -> Vec<Poly> {
    let f = monic(ring, f);
    let total = f.degree().expect("a block is nonzero");
    if total == degree {
        return vec![f];
    }

    let p = ring.coefficients().order();
    let one = ring.identity();
    let two = BigInt::from(2);

    for index in 1u64..u64::MAX {
        assert!(index < 1 << 40, "no splitting polynomial found");
        let a = candidate(ring, index, total);
        if a.degree().unwrap_or(0) == 0 {
            continue;
        }

        // A candidate sharing a factor outright splits f just as well.
        for separator in [a.clone(), splitter(ring, &f, &a, degree, &p, &two)] {
            let block = gcd(ring, f.clone(), separator);
            if block != one && block != f {
                let rest = f.div_rem(&block).0;
                let mut factors = equal_degree(ring, &block, degree);
                factors.extend(equal_degree(ring, &rest, degree));
                return factors;
            }
        }
    }
    unreachable!("the candidate loop asserts before it runs out")
}

/// The value whose gcd with `f` splits it, by whichever of the two constructions the
/// characteristic allows.
fn splitter(ring: &Ring, f: &Poly, a: &Poly, degree: usize, p: &BigInt, two: &BigInt) -> Poly {
    let quotient = ring.quotient(f.clone());
    let a = quotient.element(a.clone());
    if p == two {
        let mut trace = a.clone();
        let mut term = a;
        for _ in 1..degree {
            term = term.pow(2u32);
            trace += term.clone();
        }
        trace.representative().clone()
    } else {
        let exponent = (p.pow(degree as u32) - 1) / two;
        a.pow(exponent).representative().clone() - ring.identity()
    }
}

/// `x^(p^d) mod f`, by applying the Frobenius map `d` times rather than exponentiating
/// to `p^d` in one go.
fn frobenius(ring: &Ring, f: &Poly, d: usize) -> Poly {
    let p = ring.coefficients().order();
    let quotient: Arc<QuotientRing<Ring>> = ring.quotient(f.clone());
    let mut power = quotient.element(ring.indeterminate());
    for _ in 0..d {
        power = power.pow(p.clone());
    }
    power.representative().clone()
}

/// The factors of `g`, with every multiplicity multiplied by `p`.
fn raise(ring: &Ring, g: &Poly) -> Vec<(Poly, u32)> {
    let p: u32 = ring
        .coefficients()
        .order()
        .try_into()
        .expect("p fits a machine integer");
    squarefree(ring, g)
        .into_iter()
        .map(|(q, multiplicity)| (q, multiplicity * p))
        .collect()
}

/// `g` with `g(x)^p == f(x)`, for an `f` that is a polynomial in `x^p`.
///
/// Over a prime field the Frobenius map fixes the coefficients, so taking the root is
/// only a matter of reading off every `p`-th one.
fn pth_root(ring: &Ring, f: &Poly) -> Poly {
    let p: usize = ring
        .coefficients()
        .order()
        .try_into()
        .expect("p fits a machine integer");
    ring.element(
        f.coefficients()
            .iter()
            .step_by(p)
            .cloned()
            .collect::<Vec<_>>(),
    )
}

/// The polynomial whose coefficients are the base-`p` digits of `index`.
fn candidate(ring: &Ring, index: u64, degree_bound: usize) -> Poly {
    let field = ring.coefficients();
    let p: u64 = field.order().try_into().expect("p fits a machine integer");
    let mut coefficients = Vec::new();
    let mut remaining = index;
    while remaining > 0 && coefficients.len() < degree_bound {
        coefficients.push(field.element((remaining % p) as i64));
        remaining /= p;
    }
    ring.element(coefficients)
}

fn monic(ring: &Ring, f: &Poly) -> Poly {
    let field = ring.coefficients();
    let lead = f.lead().expect("the zero polynomial has no monic form");
    let inverse = field
        .invert(lead)
        .expect("a leading coefficient is nonzero");
    ring.element(
        f.coefficients()
            .iter()
            .map(|c| inverse.clone() * c.clone())
            .collect::<Vec<_>>(),
    )
}

fn gcd(ring: &Ring, a: Poly, b: Poly) -> Poly {
    extended_gcd(ring, a, b).0
}

/// Degree first, then the coefficients as least non-negative residues.
///
/// The residues have to be canonicalised: a representative is only reduced, not
/// canonical, so `1` and `-6` are the same element of `F_7` and sorting on the
/// representatives would order the factors by which one the arithmetic happened to
/// leave behind.
fn sort_key(ring: &Ring, f: &Poly) -> (usize, Vec<BigInt>) {
    let p = ring.coefficients().order();
    let coefficients = f
        .coefficients()
        .iter()
        .map(|c| {
            let residue: BigInt = c.representative().clone().into();
            ((residue % &p) + &p) % &p
        })
        .collect();
    (f.coefficients().len(), coefficients)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::finite_fields::is_irreducible;
    use crate::integers::Integers;
    use crate::structures::Ring as _;

    fn ring_over(p: u32) -> Ring {
        use crate::polynomials::RingExt;
        Integers::modulo(p).polynomials()
    }

    fn poly(ring: &Ring, coefficients: &[i64]) -> Poly {
        let field = ring.coefficients();
        ring.element(
            coefficients
                .iter()
                .map(|c| field.element(*c))
                .collect::<Vec<_>>(),
        )
    }

    /// Deterministic coefficients in `0..p`, so the sweep is reproducible.
    fn pseudorandom(ring: &Ring, len: usize, seed: u64) -> Poly {
        let field = ring.coefficients();
        let p: u64 = field.order().try_into().unwrap();
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let mut coefficients: Vec<_> = (0..len)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                field.element(((state >> 33) % p) as i64)
            })
            .collect();
        coefficients.push(field.identity());
        ring.element(coefficients)
    }

    fn product(ring: &Ring, factors: &[(Poly, u32)]) -> Poly {
        factors.iter().fold(ring.identity(), |acc, (q, m)| {
            (0..*m).fold(acc, |acc, _| &acc * q)
        })
    }

    #[test]
    fn factors_multiply_back_and_are_each_irreducible() {
        for p in [2u32, 3, 5, 7] {
            let ring = ring_over(p);
            for degree in 1..=7usize {
                for seed in 1..=12u64 {
                    let f = pseudorandom(&ring, degree, seed * 31 + p as u64);
                    let factors = factorise(&ring, &f);

                    assert_eq!(product(&ring, &factors), f, "F_{p}, {f}");
                    for (q, multiplicity) in &factors {
                        assert!(is_irreducible(&ring, q), "F_{p}: {q} from {f}");
                        assert!(*multiplicity >= 1);
                    }
                    let total: usize = factors
                        .iter()
                        .map(|(q, m)| q.degree().unwrap() * *m as usize)
                        .sum();
                    assert_eq!(total, f.degree().unwrap(), "F_{p}, {f}");
                }
            }
        }
    }

    #[test]
    fn known_factorisations() {
        let f2 = ring_over(2);
        let x = f2.indeterminate();

        // x^2 + x = x(x + 1), and x^4 + 1 = (x + 1)^4 over F_2.
        assert_eq!(
            factorise(&f2, &(x.pow(2) + x.clone())),
            vec![(x.clone(), 1), (x.clone() + 1, 1)]
        );
        assert_eq!(factorise(&f2, &(x.pow(4) + 1)), vec![(x.clone() + 1, 4)]);

        // An irreducible comes back as itself.
        let irreducible = x.pow(4) + x.clone() + 1;
        assert!(is_irreducible(&f2, &irreducible));
        assert_eq!(factorise(&f2, &irreducible), vec![(irreducible, 1)]);

        // x^3 - 1 over F_7: the cube roots of one are 1, 2 and 4.
        let f7 = ring_over(7);
        let y = f7.indeterminate();
        // Ordered by least non-negative residue, so x - 4 comes before x - 2.
        assert_eq!(
            factorise(&f7, &(y.pow(3) - 1)),
            vec![(y.clone() - 4, 1), (y.clone() - 2, 1), (y.clone() - 1, 1)]
        );

        // A non-monic input factors into monic pieces; the leading coefficient is left
        // to the caller.
        let scaled = (y.clone() - 1) * 3;
        assert_eq!(factorise(&f7, &scaled), vec![(y.clone() - 1, 1)]);
    }

    #[test]
    fn multiplicities_divisible_by_the_characteristic() {
        // The case Yun's algorithm cannot do: the derivative of f is zero, so the
        // repeated factor is invisible to it.
        let f2 = ring_over(2);
        let x = f2.indeterminate();

        let squared = (x.pow(2) + x.clone() + 1).pow(2);
        assert_eq!(squared.derivative(), f2.zero());
        assert_eq!(
            factorise(&f2, &squared),
            vec![(x.pow(2) + x.clone() + 1, 2)]
        );

        // x^8 = x^(2^3), three p-th roots deep.
        assert_eq!(factorise(&f2, &x.pow(8)), vec![(x.clone(), 8)]);

        // Mixed: x^2 (x + 1)^4 (x^2 + x + 1).
        let mixed = x.pow(2) * (x.clone() + 1).pow(4) * (x.pow(2) + x.clone() + 1);
        assert_eq!(
            factorise(&f2, &mixed),
            vec![
                (x.clone(), 2),
                (x.clone() + 1, 4),
                (x.pow(2) + x.clone() + 1, 1)
            ]
        );

        let f3 = ring_over(3);
        let z = f3.indeterminate();
        assert_eq!(factorise(&f3, &(z.pow(3) - 1)), vec![(z.clone() - 1, 3)]);
    }

    #[test]
    fn squarefree_parts_are_squarefree_and_coprime() {
        for p in [2u32, 3, 5] {
            let ring = ring_over(p);
            for degree in 1..=6usize {
                for seed in 1..=8u64 {
                    let f = pseudorandom(&ring, degree, seed * 17 + p as u64);
                    let parts = squarefree(&ring, &f);

                    // Each part is squarefree: it shares no factor with its derivative.
                    for (part, _) in &parts {
                        let d = part.derivative();
                        if d != ring.zero() {
                            assert_eq!(gcd(&ring, part.clone(), d), ring.identity(), "{part}");
                        }
                    }
                    // Pairwise coprime.
                    for (i, (a, _)) in parts.iter().enumerate() {
                        for (b, _) in &parts[i + 1..] {
                            assert_eq!(
                                gcd(&ring, a.clone(), b.clone()),
                                ring.identity(),
                                "{a} and {b}"
                            );
                        }
                    }
                    assert_eq!(product(&ring, &parts), monic(&ring, &f), "F_{p}, {f}");
                }
            }
        }
    }

    #[test]
    fn distinct_degree_groups_by_degree() {
        let f2 = ring_over(2);
        let x = f2.indeterminate();
        // x(x + 1) has both degree-one factors, (x^2 + x + 1) is the degree-two one.
        let f = x.clone() * (x.clone() + 1) * (x.pow(2) + x.clone() + 1);
        let blocks = distinct_degree(&f2, &f);
        assert_eq!(
            blocks,
            vec![(x.pow(2) + x.clone(), 1), (x.pow(2) + x.clone() + 1, 2)]
        );
        assert_eq!(
            product(&f2, &[(blocks[0].0.clone(), 1), (blocks[1].0.clone(), 1)]),
            f
        );
    }

    /// The splitter rests on one property, which this asserts directly rather than
    /// through whether a split happened to occur.
    ///
    /// For odd `p` it is `a^((p^d - 1)/2)`, a square root of one, so modulo each
    /// irreducible factor it is `1` or `-1` - that is what makes the gcd pick out a
    /// subset of the factors. In characteristic two the trace lands in `F_2`, so modulo
    /// each factor it is `0` or `1`. Either way the splitter `s` satisfies
    /// `q | s * (s + c)` for every factor `q`, with `c` two or one.
    #[test]
    fn the_splitter_is_plus_or_minus_one_modulo_every_factor() {
        let two = BigInt::from(2);

        let f2 = ring_over(2);
        let x = f2.indeterminate();
        let (a, b) = (x.pow(3) + x.clone() + 1, x.pow(3) + x.pow(2) + 1);
        assert!(is_irreducible(&f2, &a) && is_irreducible(&f2, &b) && a != b);
        assert_splitter_is_a_root_of_one(&f2, &[a, b], 3, &two);

        let f5 = ring_over(5);
        let y = f5.indeterminate();
        let (c, d) = (y.pow(2) + 2, y.pow(2) + 3);
        assert!(is_irreducible(&f5, &c) && is_irreducible(&f5, &d) && c != d);
        assert_splitter_is_a_root_of_one(&f5, &[c, d], 2, &two);
    }

    fn assert_splitter_is_a_root_of_one(
        ring: &Ring,
        factors: &[Poly],
        degree: usize,
        two: &BigInt,
    ) {
        let p = ring.coefficients().order();
        let f = factors.iter().fold(ring.identity(), |acc, q| &acc * q);
        let shift = if p == *two {
            ring.identity()
        } else {
            ring.from_integer(2u32)
        };

        let (mut checked, mut splits) = (0, 0);
        for index in 1..=12u64 {
            let a = candidate(ring, index, f.degree().unwrap());
            if a.degree().unwrap_or(0) == 0 || gcd(ring, f.clone(), a.clone()) != ring.identity() {
                continue;
            }
            let s = splitter(ring, &f, &a, degree, &p, two);
            let product = &s * &(&s + &shift);
            for q in factors {
                assert_eq!(
                    product.div_rem(q).1,
                    ring.zero(),
                    "splitter for {a} is not a root of one modulo {q}"
                );
            }
            let block = gcd(ring, f.clone(), s);
            if block != ring.identity() && block != f {
                splits += 1;
            }
            checked += 1;
        }
        assert!(checked >= 4, "too few usable candidates to be meaningful");
        // The property above is satisfied vacuously by a splitter that is always zero,
        // which never separates anything, so the splits have to be counted too.
        assert!(splits > 0, "no candidate below 13 split {f}");
    }

    #[test]
    #[should_panic(expected = "zero polynomial")]
    fn rejects_the_zero_polynomial() {
        let f2 = ring_over(2);
        factorise(&f2, &f2.zero());
    }

    #[test]
    fn a_unit_has_no_factors() {
        let f2 = ring_over(2);
        assert!(factorise(&f2, &f2.identity()).is_empty());
        assert!(factorise(&f2, &poly(&f2, &[1])).is_empty());
    }
}
