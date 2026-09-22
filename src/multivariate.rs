//! Multivariate polynomials `R[x_0, .., x_{n-1}]`, with the variable count fixed at
//! run time rather than in the type.
//!
//! Terms are stored sparsely, keyed by exponent vector, so a polynomial in many
//! variables costs only what its nonzero terms cost.

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, Domain, Field, Monoid, Ring, RingOps, SemiRing, Semigroup,
};
use std::cmp::Ordering;
use std::collections::BTreeMap;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use std::rc::Rc;

/// The exponents of one monomial, one entry per variable.
pub type Monomial = Vec<u32>;

/// Whether `divisor` divides `m`, i.e. is no larger in every variable.
pub fn divides(divisor: &Monomial, m: &Monomial) -> bool {
    divisor.len() == m.len() && divisor.iter().zip(m).all(|(d, e)| d <= e)
}

/// `m / divisor`, by subtracting exponents.
///
/// # Panics
/// If `divisor` does not divide `m`.
pub fn divide(m: &Monomial, divisor: &Monomial) -> Monomial {
    assert!(divides(divisor, m), "monomial does not divide");
    m.iter().zip(divisor).map(|(e, d)| e - d).collect()
}

/// A total order on monomials.
///
/// Passed to the algorithms that need one rather than held by the ring: the order
/// changes what division returns, but not what the ring is. Storage stays keyed by
/// plain exponent-vector order, which only has to be canonical.
pub trait MonomialOrder: fmt::Debug {
    fn cmp(&self, a: &Monomial, b: &Monomial) -> Ordering;
}

/// Lexicographic: the first variable to differ decides.
#[derive(Debug, Clone, Copy, Default)]
pub struct Lex;

impl MonomialOrder for Lex {
    fn cmp(&self, a: &Monomial, b: &Monomial) -> Ordering {
        a.cmp(b)
    }
}

/// Total degree first, ties broken lexicographically.
#[derive(Debug, Clone, Copy, Default)]
pub struct GradedLex;

impl MonomialOrder for GradedLex {
    fn cmp(&self, a: &Monomial, b: &Monomial) -> Ordering {
        let (da, db) = (a.iter().sum::<u32>(), b.iter().sum::<u32>());
        da.cmp(&db).then_with(|| a.cmp(b))
    }
}

/// The ring `R[x_0, .., x_{n-1}]`.
#[derive(Debug)]
pub struct MultivariatePolynomialRing<R: Ring>
where
    R::E: RingOps + Eq,
{
    coeff_ring: R,
    variables: usize,
}

/// An element of [`MultivariatePolynomialRing`].
///
/// Invariant: no term has a zero coefficient, and every exponent vector has length
/// equal to the ring's variable count. Equality is therefore structural.
#[derive(Debug)]
pub struct MultivariatePolynomial<R: Ring>
where
    R::E: RingOps + Eq,
{
    terms: BTreeMap<Monomial, R::E>,
    ring: Rc<MultivariatePolynomialRing<R>>,
}

impl<R> MultivariatePolynomialRing<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    /// Construct `R[x_0, .., x_{n-1}]`.
    pub fn new(coeff_ring: R, variables: usize) -> Rc<Self> {
        Rc::new(MultivariatePolynomialRing {
            coeff_ring,
            variables,
        })
    }

    /// The coefficient ring.
    pub fn coefficients(&self) -> &R {
        &self.coeff_ring
    }

    /// How many variables this ring has.
    pub fn variables(&self) -> usize {
        self.variables
    }

    /// The variable `x_i`, as a polynomial.
    ///
    /// # Panics
    /// If `i` is not below [`Self::variables`].
    pub fn variable(self: &Rc<Self>, i: usize) -> MultivariatePolynomial<R> {
        assert!(i < self.variables, "no variable x_{i} in {} of them", self.variables);
        let mut exponents = vec![0; self.variables];
        exponents[i] = 1;
        self.element(vec![(exponents, self.coeff_ring.identity())])
    }
}

impl<R> MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    /// Terms in ascending monomial order, each an exponent vector and its coefficient.
    pub fn terms(&self) -> impl Iterator<Item = (&Monomial, &R::E)> {
        self.terms.iter()
    }

    /// The greatest term under `order`, or [`None`] for the zero polynomial.
    pub fn lead(&self, order: &impl MonomialOrder) -> Option<(&Monomial, &R::E)> {
        self.terms.iter().max_by(|(a, _), (b, _)| order.cmp(a, b))
    }

    /// Total degree, or [`None`] for the zero polynomial.
    pub fn degree(&self) -> Option<u32> {
        self.terms.keys().map(|m| m.iter().sum::<u32>()).max()
    }

    /// Evaluates at one point, by summing each term.
    ///
    /// # Panics
    /// If `point` does not have one entry per variable.
    pub fn evaluate(&self, point: &[R::E]) -> R::E {
        assert_eq!(
            point.len(),
            self.ring.variables,
            "point has the wrong number of coordinates"
        );
        self.terms
            .iter()
            .fold(self.ring.coeff_ring.zero(), |sum, (monomial, c)| {
                let term = monomial.iter().zip(point).fold(c.clone(), |acc, (&e, x)| {
                    (0..e).fold(acc, |acc, _| acc * x.clone())
                });
                sum + term
            })
    }
}

impl<R> MultivariatePolynomial<R>
where
    R: Field,
    R::E: RingOps + Eq,
{
    /// Divides by several polynomials at once, returning one quotient each and a
    /// remainder, with `self == sum q_i * divisors_i + remainder`.
    ///
    /// No term of the remainder is divisible by any divisor's leading monomial. The
    /// result depends both on `order` and on the order of `divisors`, which is why this
    /// is not a Euclidean division and the ring is not a [`crate::structures::EuclideanDomain`].
    ///
    /// # Panics
    /// If any divisor is zero.
    pub fn divide(
        &self,
        divisors: &[Self],
        order: &impl MonomialOrder,
    ) -> (Vec<Self>, Self) {
        let ring = &self.ring;
        let coeff_ring = &ring.coeff_ring;

        let leads: Vec<(Monomial, R::E)> = divisors
            .iter()
            .map(|g| {
                let (m, c) = g.lead(order).expect("division by the zero polynomial");
                (
                    m.clone(),
                    coeff_ring
                        .invert(c)
                        .expect("leading coefficient must be invertible in a field"),
                )
            })
            .collect();

        let mut quotients = vec![ring.zero(); divisors.len()];
        let mut remainder = ring.zero();
        let mut p = self.clone();

        // Each step strips the leading term of `p`, so the order decreases and this ends.
        while let Some((lead_m, lead_c)) = p.lead(order).map(|(m, c)| (m.clone(), c.clone())) {
            match leads.iter().position(|(m, _)| divides(m, &lead_m)) {
                Some(i) => {
                    let factor = ring.element(vec![(
                        divide(&lead_m, &leads[i].0),
                        lead_c * leads[i].1.clone(),
                    )]);
                    quotients[i] += factor.clone();
                    p = p - factor * divisors[i].clone();
                }
                None => {
                    let term = ring.element(vec![(lead_m, lead_c)]);
                    remainder += term.clone();
                    p = p - term;
                }
            }
        }
        (quotients, remainder)
    }
}

/// Drops zero coefficients, which is what keeps equality structural.
fn prune<R>(ring: &R, terms: &mut BTreeMap<Monomial, R::E>)
where
    R: Ring,
    R::E: RingOps + Eq,
{
    let zero = ring.zero();
    terms.retain(|_, c| *c != zero);
}

impl<R> Domain for Rc<MultivariatePolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type E = MultivariatePolynomial<R>;
    type Repr = Vec<(Monomial, R::E)>;

    /// Sums duplicate monomials and drops zero coefficients.
    ///
    /// # Panics
    /// If any exponent vector's length is not the ring's variable count.
    fn element<T: Into<Self::Repr>>(&self, value: T) -> Self::E {
        let mut terms: BTreeMap<Monomial, R::E> = BTreeMap::new();
        for (monomial, c) in value.into() {
            assert_eq!(
                monomial.len(),
                self.variables,
                "monomial has the wrong number of exponents"
            );
            match terms.remove(&monomial) {
                Some(existing) => {
                    terms.insert(monomial, existing + c);
                }
                None => {
                    terms.insert(monomial, c);
                }
            }
        }
        prune(&self.coeff_ring, &mut terms);
        MultivariatePolynomial {
            terms,
            ring: Rc::clone(self),
        }
    }
}

impl<R> Clone for MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn clone(&self) -> Self {
        MultivariatePolynomial {
            terms: self.terms.clone(),
            ring: Rc::clone(&self.ring),
        }
    }
}

impl<R> PartialEq for MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        debug_assert!(
            Rc::ptr_eq(&self.ring, &other.ring),
            "MultivariatePolynomial operands belong to different rings"
        );
        self.terms == other.terms
    }
}

impl<R> Eq for MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Add for MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        debug_assert!(Rc::ptr_eq(&self.ring, &rhs.ring), "different rings");
        for (monomial, c) in rhs.terms {
            match self.terms.remove(&monomial) {
                Some(existing) => {
                    self.terms.insert(monomial, existing + c);
                }
                None => {
                    self.terms.insert(monomial, c);
                }
            }
        }
        prune(&self.ring.coeff_ring, &mut self.terms);
        self
    }
}

impl<R> Sub for MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type Output = Self;
    fn sub(mut self, rhs: Self) -> Self::Output {
        debug_assert!(Rc::ptr_eq(&self.ring, &rhs.ring), "different rings");
        let zero = self.ring.coeff_ring.zero();
        for (monomial, c) in rhs.terms {
            match self.terms.remove(&monomial) {
                Some(existing) => {
                    self.terms.insert(monomial, existing - c);
                }
                None => {
                    self.terms.insert(monomial, zero.clone() - c);
                }
            }
        }
        prune(&self.ring.coeff_ring, &mut self.terms);
        self
    }
}

impl<R> Mul for MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        debug_assert!(Rc::ptr_eq(&self.ring, &rhs.ring), "different rings");
        let mut terms: BTreeMap<Monomial, R::E> = BTreeMap::new();
        for (a, x) in &self.terms {
            for (b, y) in &rhs.terms {
                // Multiplying monomials adds their exponents.
                let monomial: Monomial = a.iter().zip(b).map(|(i, j)| i + j).collect();
                let product = x.clone() * y.clone();
                match terms.remove(&monomial) {
                    Some(existing) => {
                        terms.insert(monomial, existing + product);
                    }
                    None => {
                        terms.insert(monomial, product);
                    }
                }
            }
        }
        prune(&self.ring.coeff_ring, &mut terms);
        MultivariatePolynomial {
            terms,
            ring: Rc::clone(&self.ring),
        }
    }
}

/// Compound assignment, swapping in the zero polynomial so the left operand is not
/// cloned. Written directly rather than forwarding, so neither carries a binder.
macro_rules! multivariate_assign_ops {
    ($($op:ident, $method:ident, $base_method:ident);* $(;)?) => {$(
        impl<R> $op for MultivariatePolynomial<R>
        where
            R: Ring,
            R::E: RingOps + Eq,
        {
            fn $method(&mut self, rhs: Self) {
                let placeholder = MultivariatePolynomial {
                    terms: BTreeMap::new(),
                    ring: Rc::clone(&self.ring),
                };
                let lhs = std::mem::replace(self, placeholder);
                *self = lhs.$base_method(rhs);
            }
        }
    )*};
}
multivariate_assign_ops!(AddAssign, add_assign, add; MulAssign, mul_assign, mul);

impl<R> Semigroup for Rc<MultivariatePolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Monoid for Rc<MultivariatePolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn identity(&self) -> Self::E {
        self.element(vec![(vec![0; self.variables], self.coeff_ring.identity())])
    }
}

impl<R> CommutativeMonoid for Rc<MultivariatePolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    fn zero(&self) -> Self::E {
        self.element(Vec::new())
    }
}

impl<R> AdditiveGroup for Rc<MultivariatePolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> SemiRing for Rc<MultivariatePolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> Ring for Rc<MultivariatePolynomialRing<R>>
where
    R: Ring,
    R::E: RingOps + Eq,
{
}

impl<R> fmt::Display for MultivariatePolynomial<R>
where
    R: Ring,
    R::E: RingOps + Eq + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.terms.is_empty() {
            return write!(f, "0");
        }
        for (n, (monomial, c)) in self.terms.iter().enumerate() {
            if n > 0 {
                write!(f, " + ")?;
            }
            let c = c.to_string();
            if c.contains(' ') {
                write!(f, "({})", c)?;
            } else {
                write!(f, "{}", c)?;
            }
            for (i, &e) in monomial.iter().enumerate() {
                match e {
                    0 => {}
                    1 => write!(f, "*x_{i}")?,
                    e => write!(f, "*x_{i}^{{{e}}}")?,
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::{Integer, Integers};
    use crate::polynomials::RingExt;

    fn int(n: i64) -> Integer {
        Integer::from(n)
    }

    /// `Z[x_0, x_1, x_2]`, the ring from the motivating example.
    fn zxyz() -> Rc<MultivariatePolynomialRing<Integers>> {
        Integers::default().multi_polynomials(3)
    }

    #[test]
    fn arithmetic_combines_like_monomials() {
        let r = zxyz();
        let (x, y, z) = (r.variable(0), r.variable(1), r.variable(2));

        assert_eq!(r.variables(), 3);
        assert_eq!(r.zero().degree(), None);
        assert_eq!(r.identity().degree(), Some(0));
        assert_eq!((x.clone() * y.clone() * z.clone()).degree(), Some(3));

        // x + x = 2x, and x - x collapses to zero rather than a zero-coefficient term.
        let two_x = r.element(vec![(vec![1, 0, 0], int(2))]);
        assert_eq!(x.clone() + x.clone(), two_x);
        assert_eq!(x.clone() - x.clone(), r.zero());
        assert_eq!((x.clone() - x.clone()).terms().count(), 0);

        // (x + y)(x - y) = x^2 - y^2
        let expected = r.element(vec![(vec![2, 0, 0], int(1)), (vec![0, 2, 0], int(-1))]);
        assert_eq!((x.clone() + y.clone()) * (x.clone() - y.clone()), expected);

        // Multiplication adds exponents.
        assert_eq!(
            x.clone() * x.clone() * y.clone(),
            r.element(vec![(vec![2, 1, 0], int(1))])
        );

        // Ring axioms on a few mixed elements.
        let p = x.clone() + y.clone() * z.clone();
        let q = r.element(vec![(vec![0, 0, 0], int(3))]) - z.clone();
        assert_eq!(p.clone() * q.clone(), q.clone() * p.clone());
        assert_eq!(p.clone() + r.zero(), p);
        assert_eq!(p.clone() * r.identity(), p);
        assert_eq!(p.clone() - p.clone(), r.zero());
    }

    #[test]
    fn evaluation_agrees_with_substitution() {
        let r = zxyz();
        let (x, y, z) = (r.variable(0), r.variable(1), r.variable(2));

        // p = 2 + x + 3*y^2 + x*y*z
        let p = r.element(vec![
            (vec![0, 0, 0], int(2)),
            (vec![1, 0, 0], int(1)),
            (vec![0, 2, 0], int(3)),
            (vec![1, 1, 1], int(1)),
        ]);
        assert_eq!(p.degree(), Some(3));

        for (a, b, c) in [(0i64, 0, 0), (1, 1, 1), (2, -1, 3), (-4, 5, -2)] {
            let expected = 2 + a + 3 * b * b + a * b * c;
            assert_eq!(
                p.evaluate(&[int(a), int(b), int(c)]),
                int(expected),
                "at ({a}, {b}, {c})"
            );
        }

        // Variables evaluate to their coordinate.
        assert_eq!(x.evaluate(&[int(7), int(8), int(9)]), int(7));
        assert_eq!(y.evaluate(&[int(7), int(8), int(9)]), int(8));
        assert_eq!(z.evaluate(&[int(7), int(8), int(9)]), int(9));
    }

    #[test]
    fn monomial_order_decides_the_leading_term() {
        // x_0^2 vs x_0*x_1^2: lex prefers the higher power of x_0, graded the higher
        // total degree. Same ring, same polynomial - only the order passed in differs.
        let r = Integers::modulo(7).multi_polynomials(2);
        let c = r.coefficients().identity();
        let p = r.element(vec![(vec![2, 0], c.clone()), (vec![1, 2], c)]);

        assert_eq!(p.lead(&Lex).unwrap().0, &vec![2, 0]);
        assert_eq!(p.lead(&GradedLex).unwrap().0, &vec![1, 2]);
        assert_eq!(r.zero().lead(&Lex), None);
    }

    #[test]
    fn monomial_divisibility() {
        assert!(divides(&vec![1, 0, 2], &vec![3, 1, 2]));
        assert!(!divides(&vec![1, 2, 0], &vec![3, 1, 2]));
        assert!(divides(&vec![0, 0, 0], &vec![3, 1, 2]));
        assert_eq!(divide(&vec![3, 1, 2], &vec![1, 0, 2]), vec![2, 1, 0]);
    }

    #[test]
    fn division_satisfies_the_quotient_remainder_identity() {
        let f7 = Integers::modulo(7);
        let r = f7.multi_polynomials(2);
        let mono = |e: Vec<u32>, c: i64| (e, f7.element(c));

        // x^2 - y^2 divided by x - y is exactly x + y.
        let f = r.element(vec![mono(vec![2, 0], 1), mono(vec![0, 2], -1)]);
        let g = r.element(vec![mono(vec![1, 0], 1), mono(vec![0, 1], -1)]);
        let (quotients, rem) = f.divide(std::slice::from_ref(&g), &Lex);
        let q = quotients.into_iter().next().unwrap();
        assert_eq!(q, r.element(vec![mono(vec![1, 0], 1), mono(vec![0, 1], 1)]));
        assert_eq!(rem, r.zero());
        assert_eq!(q * g.clone() + rem, f);

        // Cox, Little & O'Shea's worked example: x^2*y + x*y^2 + y^2 by (xy - 1, y^2 - 1)
        // gives quotients (x + y, 1) and remainder x + y + 1 under lex.
        let f = r.element(vec![
            mono(vec![2, 1], 1),
            mono(vec![1, 2], 1),
            mono(vec![0, 2], 1),
        ]);
        let g1 = r.element(vec![mono(vec![1, 1], 1), mono(vec![0, 0], -1)]);
        let g2 = r.element(vec![mono(vec![0, 2], 1), mono(vec![0, 0], -1)]);
        let (quotients, remainder) = f.divide(&[g1.clone(), g2.clone()], &Lex);
        assert_eq!(
            quotients[0],
            r.element(vec![mono(vec![1, 0], 1), mono(vec![0, 1], 1)])
        );
        assert_eq!(quotients[1], r.element(vec![mono(vec![0, 0], 1)]));
        assert_eq!(
            remainder,
            r.element(vec![
                mono(vec![1, 0], 1),
                mono(vec![0, 1], 1),
                mono(vec![0, 0], 1)
            ])
        );

        // The identity holds, and no remainder term is divisible by a leading monomial.
        let reconstructed =
            quotients[0].clone() * g1.clone() + quotients[1].clone() * g2.clone() + remainder.clone();
        assert_eq!(reconstructed, f);
        for (m, _) in remainder.terms() {
            assert!(!divides(g1.lead(&Lex).unwrap().0, m));
            assert!(!divides(g2.lead(&Lex).unwrap().0, m));
        }

        // Dividing by a constant scales; dividing something smaller leaves it whole.
        let three = r.element(vec![mono(vec![0, 0], 3)]);
        let (q3, _) = f.divide(std::slice::from_ref(&three), &Lex);
        assert_eq!(q3[0].clone() * three, f);

        // Nothing divisible: the dividend falls through to the remainder.
        let x = r.variable(0);
        let y_squared = r.element(vec![mono(vec![0, 2], 1)]);
        let (qs, rem) = x.divide(&[y_squared], &Lex);
        assert_eq!((qs[0].clone(), rem), (r.zero(), x.clone()));

        // A different order can give a different decomposition; the identity still holds.
        let (qs, rem) = f.divide(&[g1.clone(), g2.clone()], &GradedLex);
        assert_eq!(qs[0].clone() * g1 + qs[1].clone() * g2 + rem, f);
    }

    #[test]
    fn display_names_each_variable() {
        let r = zxyz();
        let (x, y) = (r.variable(0), r.variable(1));
        assert_eq!(format!("{}", r.zero()), "0");
        assert_eq!(format!("{}", r.identity()), "1");
        assert_eq!(format!("{}", x.clone() * x.clone() * y), "1*x_0^{2}*x_1");
    }

    #[test]
    #[should_panic(expected = "wrong number of exponents")]
    fn rejects_a_monomial_of_the_wrong_arity() {
        zxyz().element(vec![(vec![1, 0], int(1))]);
    }
}
