//! Finite fields `F_{p^k}`, as `F_p[x]` modulo an irreducible polynomial of degree `k`.
//!
//! The prime case needs none of this: `F_p` is [`Integers::modulo`] with a prime
//! modulus. What an extension needs is an irreducible polynomial to quotient by, which
//! this module either looks up or finds.

use crate::integers::{Integer, Integers};
use crate::number_theory::{factorise, gcd};
use crate::polynomials::{Polynomial, PolynomialRing, RingExt};
use crate::structures::{
    Domain, EuclideanDomain, Field, Monoid, QuotientRing, QuotientRingElement, RingOps,
};
use num_bigint::BigInt;
use num_traits::{One, Zero};
use std::ops::MulAssign;
use std::sync::Arc;

/// `F_p`, the integers modulo a prime.
pub type PrimeField = Arc<QuotientRing<Integers>>;

/// An element of a prime field.
pub type Residue = QuotientRingElement<Integers>;

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

/// A field with finitely many elements, which knows how many and can list them.
///
/// The order is what square roots are stated in terms of, not the characteristic, so
/// one implementation covers `F_p` and its extensions alike.
pub trait FiniteField: Field
where
    Self::E: RingOps + Eq,
{
    /// How many elements the field has.
    fn order(&self) -> BigInt;

    /// The characteristic `p`, where the order is a power of it.
    fn characteristic(&self) -> BigInt;

    /// The element at `index`, which must be below [`FiniteField::order`]. Any fixed
    /// enumeration of the field will do.
    fn element_at(&self, index: &BigInt) -> Self::E;

    /// Whether `x` is a `k`-th power. See [`is_kth_power`], which this forwards to.
    fn is_kth_power(&self, x: &Self::E, k: u32) -> bool
    where
        Self: Sized,
        Self::E: for<'a> MulAssign<&'a Self::E>,
    {
        crate::finite_fields::is_kth_power(self, x, k)
    }

    /// The Legendre symbol of `x`. See [`legendre_symbol`], which this forwards to.
    fn legendre_symbol(&self, x: &Self::E) -> i8
    where
        Self: Sized,
        Self::E: for<'a> MulAssign<&'a Self::E>,
    {
        crate::finite_fields::legendre_symbol(self, x)
    }

    /// A square root of `x`. See [`sqrt`], which this forwards to.
    fn sqrt(&self, x: &Self::E) -> Option<Self::E>
    where
        Self: Sized,
        Self::E: for<'a> MulAssign<&'a Self::E>,
    {
        crate::finite_fields::sqrt(self, x)
    }

    /// An `r`-th root of `x` for a prime `r`. See [`root`], which this forwards to.
    fn root(&self, x: &Self::E, r: u32) -> Option<Self::E>
    where
        Self: Sized,
        Self::E: for<'a> MulAssign<&'a Self::E>,
    {
        crate::finite_fields::root(self, x, r)
    }

    /// A primitive `r`-th root of unity. See [`root_of_unity`], which this forwards to.
    fn root_of_unity(&self, r: u32) -> Option<Self::E>
    where
        Self: Sized,
        Self::E: for<'a> MulAssign<&'a Self::E>,
    {
        crate::finite_fields::root_of_unity(self, r)
    }
}

impl FiniteField for PrimeField {
    fn order(&self) -> BigInt {
        // The inherent method on the ring; `self.order()` would find this one again.
        QuotientRing::order(self)
    }

    /// The modulus, which this type assumes is prime.
    fn characteristic(&self) -> BigInt {
        QuotientRing::order(self)
    }

    fn element_at(&self, index: &BigInt) -> Residue {
        self.element(Integer::from(index.clone()))
    }
}

impl FiniteField for ExtensionField {
    fn order(&self) -> BigInt {
        let degree = self.modulus().degree().expect("the modulus is nonzero");
        self.characteristic().pow(degree as u32)
    }

    /// The characteristic of the coefficient field, which the extension shares.
    fn characteristic(&self) -> BigInt {
        self.base().coefficients().order()
    }

    /// The polynomial whose coefficients are the base-`p` digits of `index`.
    fn element_at(&self, index: &BigInt) -> QuotientRingElement<Arc<PolynomialRing<PrimeField>>> {
        let ring = self.base();
        let field = ring.coefficients();
        let p = field.order();
        let degree = self.modulus().degree().expect("the modulus is nonzero");

        let mut remaining = index.clone();
        let coefficients: Vec<_> = (0..degree)
            .map(|_| {
                let digit = field.element(Integer::from(&remaining % &p));
                remaining /= &p;
                digit
            })
            .collect();
        self.element(ring.element(coefficients))
    }
}

/// The Legendre symbol of `a`: `0` when `a` is zero, `1` when it is a nonzero square,
/// and `-1` when it is not a square.
///
/// # Panics
/// If the field has even order, where every element is a square and the symbol says
/// nothing.
pub fn legendre_symbol<F>(field: &F, a: &F::E) -> i8
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    let q = field.order();
    assert!(
        !(&q % 2u32).is_zero(),
        "the Legendre symbol needs an odd order"
    );
    if field.is_zero(a) {
        return 0;
    }
    // Euler's criterion: for nonzero a, a^((q-1)/2) is 1 if a is a square and -1 if not.
    if field.pow(a, (&q - 1u32) / 2u32) == field.identity() {
        1
    } else {
        -1
    }
}

/// Whether `x` is a `k`-th power in the field.
///
/// No root is computed: the `k`-th powers are the kernel of raising to
/// `(q-1)/gcd(k, q-1)`, so one exponentiation answers it. Zero is a `k`-th power.
///
/// # Panics
/// If `k` is zero.
pub fn is_kth_power<F>(field: &F, x: &F::E, k: u32) -> bool
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    assert!(k > 0, "a zeroth power is not a useful question");
    if field.is_zero(x) {
        return true;
    }
    let order = field.order() - 1u32;
    let exponent = &order / gcd(BigInt::from(k), order.clone());
    field.pow(x, exponent) == field.identity()
}

/// A square root of `a`, or [`None`] when `a` is not a square.
///
/// Both roots are `r` and `-r`; this returns whichever the algorithm reaches. Tonelli
/// and Shanks' algorithm is [`root`] at `r = 2`, which this forwards to.
///
/// `F` must really be a field. A quotient by a composite modulus is accepted by the
/// type and the answer is then unspecified: `Z/15` reports that 4 is not a square,
/// though `2^2 = 4` there. The assertions below catch some such cases but not all.
pub fn sqrt<F>(field: &F, a: &F::E) -> Option<F::E>
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    root(field, a, 2)
}

/// An `r`-th root of `a`, or [`None`] when `a` is not an `r`-th power, by Adleman,
/// Manders and Miller's algorithm.
///
/// There are `r` roots when `r` divides `q - 1` and one otherwise. This returns
/// whichever the algorithm reaches, and [`root_of_unity`] is the factor between them.
///
/// # Panics
/// If `r` is not prime, where one root at a time is not the whole question.
pub fn root<F>(field: &F, a: &F::E, r: u32) -> Option<F::E>
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    assert_prime(r);
    if field.is_zero(a) {
        return Some(field.zero());
    }
    let one = field.identity();
    let order = field.order() - 1u32;
    if order.is_one() {
        // F_2, whose only nonzero element is its own root.
        return Some(a.clone());
    }

    // q - 1 = r^s * t, with r dividing t no further.
    let (mut t, mut s) = (order.clone(), 0u32);
    while (&t % r).is_zero() {
        t /= r;
        s += 1;
    }

    // With r coprime to q - 1, raising to the r is a bijection and the root is the
    // inverse exponent. The characteristic always lands here.
    if s == 0 {
        return Some(field.pow(a, inverse_modulo(r, &order)));
    }

    // Euler's criterion for r-th powers.
    if field.pow(a, &order / r) != one {
        return None;
    }

    // d * r = 1 + k * t, so a^d is a root but for a factor in the Sylow r-subgroup.
    let d = if t.is_one() {
        BigInt::one()
    } else {
        inverse_modulo(r, &t)
    };
    let k = (&d * r - 1u32) / &t;
    let mut root = field.pow(a, d);
    if s == 1 {
        // The criterion says a^t = 1, which leaves that factor trivial.
        return Some(root);
    }

    // c generates the Sylow r-subgroup, of order r^s, and zeta the r-th roots of unity
    // within it. From here on root^r = a * h, with h in that subgroup.
    let mut h = field.pow(&field.pow(a, t.clone()), k);
    let c = field.pow(&non_power(field, r), t);
    let zeta = field.pow(&c, BigInt::from(r).pow(s - 1));
    let roots_of_unity: Vec<F::E> = (0..r).map(|j| field.pow(&zeta, j)).collect();

    // Each round divides the order of h by r, and the criterion leaves it dividing
    // r^(s-1), so this is every round there can be.
    for _ in 1..s {
        if h == one {
            break;
        }

        // The order of h, as a power of r.
        let (mut i, mut power) = (0u32, h.clone());
        while power != one {
            power = field.pow(&power, r);
            i += 1;
            assert!(i < s, "the order is not a field's");
        }

        // h's leading digit in the subgroup, as a power of zeta, which is not the
        // zeroth: i is the exact order, so this is not one.
        let leading = field.pow(&h, BigInt::from(r).pow(i - 1));
        let j = roots_of_unity
            .iter()
            .position(|power| *power == leading)
            .expect("a power of zeta");
        // Cancelling it divides the order of h by r, so this ends within s rounds.
        let correction = field.pow(
            &c,
            BigInt::from(r - j as u32) * BigInt::from(r).pow(s - i - 1),
        );
        h *= field.pow(&correction, r);
        root *= correction;
    }
    assert!(
        h == one,
        "the Sylow walk did not finish: the order is not a field's"
    );
    Some(root)
}

/// A primitive `r`-th root of unity, or [`None`] when the field holds none: `F*` is
/// cyclic of order `q - 1`, so one exists exactly when `r` divides `q - 1`.
///
/// # Panics
/// If `r` is not prime, where a non-`r`-th power need not be primitive.
pub fn root_of_unity<F>(field: &F, r: u32) -> Option<F::E>
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    assert_prime(r);
    let order = field.order() - 1u32;
    if !(&order % r).is_zero() {
        return None;
    }
    // A non-r-th power raised to (q - 1)/r is an r-th root of unity other than one,
    // and r being prime leaves nothing between that and primitive.
    Some(field.pow(&non_power(field, r), &order / r))
}

/// An element that is not an `r`-th power, by trying every one in turn.
///
/// # Panics
/// If there is none, which is every field where `r` does not divide `q - 1`.
fn non_power<F>(field: &F, r: u32) -> F::E
where
    F: FiniteField,
    F::E: RingOps + Eq + for<'x> MulAssign<&'x F::E>,
{
    let q = field.order();
    let exponent = (&q - 1u32) / r;
    // Zero and one are r-th powers, so the search starts past them.
    let mut index = BigInt::from(2);
    loop {
        assert!(index < q, "every element is an r-th power");
        let candidate = field.element_at(&index);
        if field.pow(&candidate, exponent.clone()) != field.identity() {
            return candidate;
        }
        index += 1u32;
    }
}

/// `r` inverted modulo `modulus`, as a representative in `1..modulus`.
fn inverse_modulo(r: u32, modulus: &BigInt) -> BigInt {
    let ring = Integers::modulo(modulus.clone());
    let inverse: BigInt = ring
        .inverse(Integer::from(BigInt::from(r)))
        .expect("r is coprime to the modulus")
        .representative()
        .clone()
        .into();
    ((inverse % modulus) + modulus) % modulus
}

/// # Panics
/// Unless `r` is prime.
fn assert_prime(r: u32) {
    assert!(
        factorise(r) == [(BigInt::from(r), 1)],
        "the exponent {r} is not prime"
    );
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
        if ring.gcd(difference, f.clone()) != ring.identity() {
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
    fn the_legendre_symbol_is_zero_one_or_minus_one_and_multiplicative() {
        for prime in [3u32, 5, 7, 11, 13, 17, 97] {
            let field = Integers::modulo(prime);
            let squares: Vec<_> = (0..prime)
                .map(|n| {
                    let x = field.element(n);
                    &x * &x
                })
                .collect();

            for n in 0..prime {
                let a = field.element(n);
                let symbol = field.legendre_symbol(&a);
                let expected = if field.is_zero(&a) {
                    0
                } else if squares.contains(&a) {
                    1
                } else {
                    -1
                };
                assert_eq!(symbol, expected, "({n} / {prime})");
            }

            // Multiplicative, which is the property that characterises it.
            for m in 0..prime {
                for n in 0..prime {
                    let (x, y) = (field.element(m), field.element(n));
                    assert_eq!(
                        field.legendre_symbol(&(&x * &y)),
                        field.legendre_symbol(&x) * field.legendre_symbol(&y),
                        "({m} * {n} / {prime})"
                    );
                }
            }
        }

        // 3^2 = 2 mod 7, while 3 is not a square there.
        let f7 = Integers::modulo(7);
        assert_eq!(f7.legendre_symbol(&f7.element(2)), 1);
        assert_eq!(f7.legendre_symbol(&f7.element(3)), -1);
        assert_eq!(f7.legendre_symbol(&f7.zero()), 0);
    }

    #[test]
    #[should_panic(expected = "odd order")]
    fn the_legendre_symbol_needs_an_odd_order() {
        let f2 = Integers::modulo(2);
        f2.legendre_symbol(&f2.identity());
    }

    #[test]
    fn square_roots_square_back_and_exist_exactly_for_the_residues() {
        for prime in [2u32, 3, 5, 7, 11, 13, 17, 23, 97, 257] {
            // Tonelli and Shanks needs a prime, so the test says so rather than
            // assuming it.
            assert_eq!(factorise(prime), vec![(BigInt::from(prime), 1)], "{prime}");
            let field = Integers::modulo(prime);

            let mut roots = 0;
            for n in 0..prime {
                let a = field.element(n);
                match field.sqrt(&a) {
                    Some(r) => {
                        assert_eq!(&r * &r, a, "sqrt({n}) in F_{prime}");
                        roots += 1;
                    }
                    // Not a square: nothing squares to it.
                    None => {
                        assert!(
                            (0..prime).all(|m| {
                                let x = field.element(m);
                                &x * &x != a
                            }),
                            "F_{prime} has a root of {n} that sqrt missed"
                        );
                    }
                }
            }
            // Zero, and half the nonzero elements.
            let expected = if prime == 2 { 2 } else { 1 + (prime - 1) / 2 };
            assert_eq!(roots, expected, "in F_{prime}");
        }
    }

    /// Every element of `field`, by walking its own enumeration.
    fn elements<F>(field: &F) -> Vec<F::E>
    where
        F: FiniteField,
        F::E: RingOps + Eq,
    {
        let order: u32 = field.order().try_into().expect("a small field");
        (0..order)
            .map(|i| field.element_at(&BigInt::from(i)))
            .collect()
    }

    /// Checks `sqrt` against the squares of the whole field.
    fn roots_agree_with_squaring<F>(field: &F, name: &str)
    where
        F: FiniteField,
        F::E: RingOps + Eq + std::fmt::Debug + for<'x> MulAssign<&'x F::E>,
    {
        let all = elements(field);
        // The enumeration has to be a bijection for that to mean anything.
        for (i, a) in all.iter().enumerate() {
            assert!(!all[i + 1..].contains(a), "{name} lists an element twice");
        }
        let squares: Vec<F::E> = all.iter().map(|x| x.clone() * x.clone()).collect();

        for a in &all {
            match sqrt(field, a) {
                Some(r) => assert_eq!(r.clone() * r, *a, "{name}"),
                None => assert!(!squares.contains(a), "{name} missed a root"),
            }
        }
    }

    #[test]
    fn square_roots_in_extension_fields() {
        // F_9 and F_25: q - 1 is divisible by 8, so neither shortcut applies and the
        // main loop always runs - which is why the algorithm had to stop assuming F_p.
        for (p, k) in [(3u32, 2usize), (5, 2), (3, 3)] {
            let field = finite_field(p, k);
            assert_eq!(field.order(), BigInt::from(p).pow(k as u32));
            roots_agree_with_squaring(&field, &format!("F_{p}^{k}"));
        }

        // Characteristic two: squaring is a bijection, so every element has exactly one
        // root and none is ever rejected.
        for k in [1usize, 2, 3, 4] {
            let field = finite_field(2, k);
            for a in elements(&field) {
                let r = field.sqrt(&a).expect("every element is a square");
                assert_eq!(r.clone() * r, a, "F_2^{k}");
            }
        }

        // And the prime fields still go through the same code.
        for p in [3u32, 7, 13, 17] {
            roots_agree_with_squaring(&Integers::modulo(p), &format!("F_{p}"));
        }
    }

    #[test]
    fn square_roots_in_a_field_with_a_long_two_adic_tail() {
        // 40961 = 5 * 2^13 + 1, so the main loop runs rather than the p = 3 mod 4
        // shortcut, and runs deep.
        let prime = 40961u32;
        assert_eq!(factorise(prime), vec![(BigInt::from(prime), 1)]);
        let field = Integers::modulo(prime);

        let (mut squares, mut not) = (0, 0);
        for n in (0..prime).step_by(37) {
            let a = field.element(n);
            match field.sqrt(&a) {
                Some(r) => {
                    assert_eq!(&r * &r, a, "sqrt({n})");
                    squares += 1;
                }
                None => not += 1,
            }
        }
        // Roughly half either way, which says the criterion is not answering blindly.
        assert!(squares > 100 && not > 100, "{squares} squares, {not} not");

        // A square has both roots, and they are negatives of each other.
        let a = field.element(1234) * field.element(1234);
        let r = field.sqrt(&a).expect("a square has a root");
        assert_eq!(&r * &r, a);
        assert_eq!(&(-r.clone()) * &(-r.clone()), a);
        assert_ne!(r.clone(), -r);
    }

    /// Checks `root` against the `r`-th powers of the whole field, and against
    /// `is_kth_power`, which answers the same existence question another way.
    fn roots_agree_with_powering<F>(field: &F, r: u32, name: &str)
    where
        F: FiniteField,
        F::E: RingOps + Eq + std::fmt::Debug + for<'x> MulAssign<&'x F::E>,
    {
        let all = elements(field);
        let powers: Vec<F::E> = all.iter().map(|x| field.pow(x, r)).collect();

        for a in &all {
            match root(field, a, r) {
                Some(x) => {
                    assert_eq!(field.pow(&x, r), *a, "{name}, r = {r}");
                    assert!(field.is_kth_power(a, r), "{name}, r = {r}: not a power");
                }
                None => {
                    assert!(!powers.contains(a), "{name} missed a root, r = {r}");
                    assert!(!field.is_kth_power(a, r), "{name}, r = {r}");
                }
            }
        }
    }

    #[test]
    fn prime_roots_power_back_and_exist_exactly_for_the_powers() {
        for prime in [2u32, 3, 5, 7, 11, 13, 17, 19, 23, 31, 101] {
            assert_eq!(factorise(prime), vec![(BigInt::from(prime), 1)], "{prime}");
            let field = Integers::modulo(prime);
            for r in [2u32, 3, 5, 7] {
                roots_agree_with_powering(&field, r, &format!("F_{prime}"));
            }
        }
    }

    #[test]
    fn prime_roots_in_extension_fields() {
        for (p, k) in [(3u32, 2usize), (5, 2), (7, 2), (2, 3), (2, 4), (3, 3)] {
            let field = finite_field(p, k);
            for r in [2u32, 3, 5] {
                roots_agree_with_powering(&field, r, &format!("F_{p}^{k}"));
            }
        }
    }

    #[test]
    fn prime_roots_where_the_sylow_subgroup_is_not_cyclic_of_order_r() {
        // r^2 divides q - 1 in each of these, which is the only case where the walk
        // through the Sylow subgroup runs at all: everything else is one of the two
        // shortcuts.
        for (prime, r) in [(17u32, 2u32), (19, 3), (73, 3), (101, 5), (41, 2)] {
            let order = BigInt::from(prime) - 1u32;
            let mut s = 0u32;
            let mut t = order.clone();
            while (&t % r).is_zero() {
                t /= r;
                s += 1;
            }
            assert!(s >= 2, "F_{prime} has only r^{s} in its order, r = {r}");
            roots_agree_with_powering(&Integers::modulo(prime), r, &format!("F_{prime}"));
        }
    }

    #[test]
    fn the_characteristic_root_is_unique_because_frobenius_is_a_bijection() {
        for (p, k) in [(3u32, 2usize), (5, 2), (2, 4), (7, 1)] {
            let field = finite_field(p, k);
            let all = elements(&field);
            let powers: Vec<_> = all.iter().map(|x| field.pow(x, p)).collect();

            // Frobenius is one to one, so each element has exactly one p-th root.
            for (i, x) in powers.iter().enumerate() {
                assert!(!powers[i + 1..].contains(x), "F_{p}^{k}: x^p repeats");
            }
            for a in &all {
                let x = root(&field, a, p).expect("x -> x^p is onto");
                assert_eq!(field.pow(&x, p), *a, "F_{p}^{k}");
            }
        }
    }

    #[test]
    fn roots_of_unity_are_primitive_and_exist_exactly_when_r_divides_the_order() {
        for prime in [3u32, 5, 7, 11, 13, 17, 19, 31, 101] {
            let field = Integers::modulo(prime);
            let order = BigInt::from(prime) - 1u32;
            for r in [2u32, 3, 5, 7] {
                match root_of_unity(&field, r) {
                    Some(zeta) => {
                        assert!((&order % r).is_zero(), "F_{prime}: r = {r} does not divide");
                        assert_eq!(field.pow(&zeta, r), field.identity(), "F_{prime}");
                        // r prime, so anything but one is of order exactly r.
                        assert_ne!(zeta, field.identity(), "F_{prime}, r = {r}");
                    }
                    None => assert!(!(&order % r).is_zero(), "F_{prime}: r = {r} divides"),
                }
            }
        }
    }

    #[test]
    fn one_root_and_the_roots_of_unity_give_all_of_them() {
        // 101 - 1 = 100 = 4 * 25, so fifth roots come in fives and square roots in
        // pairs.
        let field = Integers::modulo(101);
        for r in [2u32, 5] {
            let zeta = root_of_unity(&field, r).expect("r divides 100");
            for n in 1..101 {
                let a = field.element(n);
                let Some(first) = root(&field, &a, r) else {
                    continue;
                };
                let all: Vec<_> = (0..r)
                    .map(|j| first.clone() * field.pow(&zeta, j))
                    .collect();
                for (i, x) in all.iter().enumerate() {
                    assert_eq!(field.pow(x, r), a, "a root of {n}");
                    assert!(!all[i + 1..].contains(x), "two roots of {n} coincide");
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "the exponent 4 is not prime")]
    fn a_composite_exponent_is_rejected() {
        root(&Integers::modulo(17), &Integers::modulo(17).element(1), 4);
    }

    #[test]
    #[should_panic(expected = "the exponent 1 is not prime")]
    fn a_root_of_unity_needs_a_prime_too() {
        root_of_unity(&Integers::modulo(17), 1);
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
            assert_eq!(field.characteristic(), BigInt::from(p), "F_{p}^{k}");
            assert_eq!(field.order(), BigInt::from(p).pow(k as u32), "F_{p}^{k}");
        }

        // A prime field is its own characteristic.
        for p in [2u32, 7, 101] {
            let field = Integers::modulo(p);
            assert_eq!(field.characteristic(), BigInt::from(p));
            assert_eq!(FiniteField::order(&field), BigInt::from(p));
        }
    }
}
