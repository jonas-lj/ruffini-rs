//! The extended Euclidean algorithm, over any [`EuclideanDomain`].

use crate::structures::{DivRem, EuclideanDomain, RingOps};
use std::ops::MulAssign;

/// Returns `(gcd, x, y)` with `x * a + y * b == gcd`.
///
/// The gcd is canonicalised through [`EuclideanDomain::unit_part`], so coprime inputs
/// always give `gcd == identity`: positive over `Z`, monic over `F[x]`.
pub fn extended_gcd<D>(domain: &D, a: D::E, b: D::E) -> (D::E, D::E, D::E)
where
    D: EuclideanDomain,
    D::E: RingOps + DivRem + Eq + for<'a> MulAssign<&'a D::E>,
{
    let (mut old_r, mut r) = (a, b);
    let (mut old_s, mut s) = (domain.identity(), domain.zero());
    let (mut old_t, mut t) = (domain.zero(), domain.identity());
    let zero = domain.zero();
    while r != zero {
        let (q, new_r) = old_r.div_rem(&r);
        old_r = std::mem::replace(&mut r, new_r);
        let mut qs = s.clone();
        qs *= &q;
        let new_s = old_s - qs;
        old_s = std::mem::replace(&mut s, new_s);
        let mut qt = t.clone();
        qt *= &q;
        let new_t = old_t - qt;
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
