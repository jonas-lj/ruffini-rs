//! The extended Euclidean algorithm, over any [`EuclideanDomain`].

use crate::structures::{DivRem, EuclideanDomain, RingOps};

/// The greatest common divisor, canonicalised through [`EuclideanDomain::unit_part`]:
/// non-negative over `Z`, monic over `F[x]`.
///
/// [`extended_gcd`] computes this too, alongside the Bezout coefficients; carrying
/// those costs two more sequences per step, so this is the one to call when they are
/// not wanted.
pub fn gcd<D>(domain: &D, a: D::E, b: D::E) -> D::E
where
    D: EuclideanDomain,
    D::E: RingOps + DivRem + Eq,
{
    let (mut previous, mut current) = (a, b);
    let zero = domain.zero();
    while current != zero {
        let next = previous.div_rem(&current).1;
        previous = std::mem::replace(&mut current, next);
    }
    let unit = domain.unit_inverse(&domain.unit_part(&previous));
    previous * unit
}

/// Returns `(gcd, x, y)` with `x * a + y * b == gcd`.
///
/// The gcd is canonicalised through [`EuclideanDomain::unit_part`], so coprime inputs
/// always give `gcd == identity`: positive over `Z`, monic over `F[x]`.
pub fn extended_gcd<D>(domain: &D, a: D::E, b: D::E) -> (D::E, D::E, D::E)
where
    D: EuclideanDomain,
    D::E: RingOps + DivRem + Eq,
{
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (domain.identity(), domain.zero());
    let (mut old_t, mut t) = (domain.zero(), domain.identity());
    let zero = domain.zero();
    while r != zero {
        let (q, new_r) = old_r.div_rem(&r);
        old_r = std::mem::replace(&mut r, new_r);
        let new_s = old_s - q.clone() * s.clone();
        old_s = std::mem::replace(&mut s, new_s);
        let new_t = old_t - q * t.clone();
        old_t = std::mem::replace(&mut t, new_t);
    }
    let u_inv = domain.unit_inverse(&domain.unit_part(&old_r));
    (old_r * u_inv.clone(), old_s * u_inv.clone(), old_t * u_inv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::{Integer, Integers};

    fn int(n: i64) -> Integer {
        Integer::from(n)
    }

    #[test]
    fn gcd_agrees_with_the_extended_one_and_is_canonical() {
        use crate::polynomials::RingExt;
        use crate::structures::{CommutativeMonoid, Monoid};

        let z = Integers::default();
        for a in -12..=12i64 {
            for b in -12..=12i64 {
                let g = gcd(&z, int(a), int(b));
                assert_eq!(g, z.extended_gcd(int(a), int(b)).0, "gcd({a}, {b})");
                if a != 0 || b != 0 {
                    // Divides both, and is the largest that does - which pins the sign
                    // too, since the candidates are positive.
                    assert_eq!(int(a).div_rem(&g).1, int(0));
                    assert_eq!(int(b).div_rem(&g).1, int(0));
                    let largest = (1..=24i64).filter(|d| a % d == 0 && b % d == 0).max();
                    assert_eq!(g, int(largest.unwrap()), "gcd({a}, {b})");
                }
            }
        }
        // The convention at zero.
        assert_eq!(gcd(&z, int(0), int(0)), int(0));

        // Over F_7[x] the canonical form is monic, not positive.
        let f7 = Integers::modulo(7);
        let ring = f7.polynomials();
        let x = ring.indeterminate();
        let g = ring.gcd((x.pow(2) - 1) * 3, (x.clone() - 1) * 5);
        assert_eq!(g, x.clone() - 1);
        assert_eq!(g.lead(), Some(&f7.identity()));
        assert_eq!(ring.gcd(x.pow(2) + 1, x.clone() - 1), ring.identity());
        assert_eq!(ring.gcd(ring.zero(), ring.zero()), ring.zero());
    }

    #[test]
    fn canonicalises_via_unit_part() {
        let z = Integers::default();

        let (g, s, t) = extended_gcd(&z, int(4), int(7));
        assert_eq!(g, int(1));
        assert_eq!(&s * &int(4) + &t * &int(7), int(1));

        // The raw Euclidean step lands on gcd = -1 here; canonicalisation fixes the sign.
        let (g, s, t) = extended_gcd(&z, int(-4), int(7));
        assert_eq!(g, int(1));
        assert_eq!(&s * &int(-4) + &t * &int(7), int(1));

        let (g, _, _) = extended_gcd(&z, int(6), int(10));
        assert_eq!(g, int(2));
    }
}
