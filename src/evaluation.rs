//! Evaluating one polynomial at many points of an arithmetic progression.

use crate::polynomials::Polynomial;
use crate::structures::{Ring, RingOps};

/// Evaluations of a polynomial at `start`, `start + step`, and so on without end.
///
/// For degree `n`, the first value costs `O(n^2)` and every one after it `n` additions
/// and no multiplication.
pub struct Evaluations<R: Ring>
where
    R::E: RingOps + Eq,
{
    /// `state[j]` is the `j`-th forward difference at the point reached so far, so
    /// `state[0]` is the value there.
    state: Vec<R::E>,
    started: bool,
}

/// The evaluations of `p` along the progression from `start` in steps of `step`.
pub fn evaluations<R>(p: &Polynomial<R>, start: &R::E, step: &R::E) -> Evaluations<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    let degree = p.degree().unwrap_or(0);
    let mut state: Vec<R::E> =
        std::iter::successors(Some(start.clone()), |x| Some(x.clone() + step.clone()))
            .take(degree + 1)
            .map(|x| p.evaluate(&x))
            .collect();

    // Differences of differences, in place and from the back, so each subtraction still
    // reads the previous round's value.
    for k in 1..state.len() {
        for j in (k..state.len()).rev() {
            let previous = state[j - 1].clone();
            state[j] = state[j].clone() - previous;
        }
    }

    Evaluations {
        state,
        started: false,
    }
}

impl<R> Iterator for Evaluations<R>
where
    R: Ring,
    R::E: RingOps + Eq,
{
    type Item = R::E;

    fn next(&mut self) -> Option<R::E> {
        if self.started {
            // Each difference absorbs the one below it. Ascending, so the value read is
            // always the one from the previous point.
            for j in 0..self.state.len() - 1 {
                let below = self.state[j + 1].clone();
                self.state[j] += below;
            }
        } else {
            self.started = true;
        }
        Some(self.state[0].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::{Integer, Integers};
    use crate::polynomials::RingExt;
    use crate::structures::{CommutativeMonoid, Domain, Monoid};

    fn int(n: i64) -> Integer {
        Integer::from(n)
    }

    #[test]
    fn agrees_with_horner_along_the_progression() {
        let z = Integers::default();
        let zx = z.polynomials();

        // Degrees either side of the trivial cases, and progressions that start and
        // step negatively as well as positively.
        for coefficients in [
            vec![],
            vec![7],
            vec![1, 2],
            vec![-3, 0, 5],
            vec![1, -2, 3, -4],
            vec![0, 0, 0, 0, 1],
            vec![5, 4, 3, 2, 1, 6],
        ] {
            let p = zx.element(coefficients.iter().map(|c| int(*c)).collect::<Vec<_>>());
            for (start, step) in [(0, 1), (1, 1), (-4, 3), (10, -2), (0, 5), (7, 0)] {
                let expected: Vec<_> = (0..12)
                    .map(|i| p.evaluate(&int(start + step * i)))
                    .collect();
                let actual: Vec<_> = evaluations(&p, &int(start), &int(step)).take(12).collect();
                assert_eq!(actual, expected, "{p} from {start} by {step}");
            }
        }
    }

    #[test]
    fn works_over_a_quotient_ring() {
        // Reduction happens at every step, so the differences have to stay in the ring.
        let f97 = Integers::modulo(97);
        let ring = f97.polynomials();
        let p = ring.element(
            [5i64, 0, 61, 2, 44]
                .iter()
                .map(|c| f97.element(*c))
                .collect::<Vec<_>>(),
        );

        let expected: Vec<_> = (0..20)
            .map(|i| p.evaluate(&f97.element(3 + 7 * i)))
            .collect();
        let actual: Vec<_> = evaluations(&p, &f97.element(3), &f97.element(7))
            .take(20)
            .collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn the_zero_and_constant_cases_never_change() {
        let z = Integers::default();
        let zx = z.polynomials();

        for p in [zx.zero(), zx.identity(), zx.constant(int(-9))] {
            let values: Vec<_> = evaluations(&p, &int(4), &int(3)).take(5).collect();
            let expected = p.evaluate(&int(0));
            assert!(values.iter().all(|v| *v == expected), "{p}");
        }
    }

    #[test]
    fn a_polynomial_is_recovered_from_its_own_evaluations() {
        // Degree d is determined by d + 1 values, so interpolating the evaluations at
        // 0, 1, .., d has to give the polynomial back.
        let f101 = Integers::modulo(101);
        let ring = f101.polynomials();
        let p = ring.element(
            [8i64, 97, 3, 64]
                .iter()
                .map(|c| f101.element(*c))
                .collect::<Vec<_>>(),
        );

        let xs: Vec<_> = (0..4).map(|i| f101.element(i)).collect();
        let ys: Vec<_> = evaluations(&p, &f101.element(0), &f101.element(1))
            .take(4)
            .collect();
        assert_eq!(crate::interpolation::interpolate(&ring, &xs, &ys), Some(p));
    }
}
