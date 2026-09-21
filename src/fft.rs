//! Radix-2 fast Fourier transform over any ring with a root of unity.

use crate::pow::pow;
use crate::structures::{Ring, RingOps};
use std::ops::MulAssign;

/// Reorders `values` so index `i` holds what was at `i` with its bits reversed.
fn bit_reverse<T>(values: &mut [T]) {
    let n = values.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            values.swap(i, j);
        }
    }
}

/// Transforms `values` in place, where `omega` is a primitive `values.len()`-th root
/// of unity. The length must be a power of two.
///
/// Whether such an `omega` exists is the caller's problem; nothing here can check it,
/// since a ring need not have a decidable equality.
pub fn fft<R>(ring: &R, values: &mut [R::E], omega: &R::E)
where
    R: Ring,
    R::E: RingOps,
{
    let n = values.len();
    assert!(n.is_power_of_two(), "fft length must be a power of two");
    if n <= 1 {
        return;
    }
    bit_reverse(values);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let step = pow(ring, omega, n / len);
        for block in values.chunks_mut(len) {
            let mut w = ring.identity();
            for j in 0..half {
                let u = block[j].clone();
                let v = block[j + half].clone() * w.clone();
                block[j] = u.clone() + v.clone();
                block[j + half] = u - v;
                w = w * step.clone();
            }
        }
        len <<= 1;
    }
}

/// The inverse of [`fft`], given `omega_inv` and `n_inv`.
///
/// Undoing the transform runs it backwards and divides by the length, so both inverses
/// have to exist — an assumption the caller carries, since a ring need not supply them.
/// Over a field, obtain them with [`Field::invert`](crate::structures::Field::invert).
pub fn inverse_fft<R>(ring: &R, values: &mut [R::E], omega_inv: &R::E, n_inv: &R::E)
where
    R: Ring,
    R::E: RingOps + for<'a> MulAssign<&'a R::E>,
{
    fft(ring, values, omega_inv);
    for v in values.iter_mut() {
        *v *= n_inv;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integers::Integers;
    use crate::structures::{Domain, Field, Monoid};

    /// The transform by its definition, at `O(n^2)`, to check [`fft`] against.
    fn naive_dft<R>(ring: &R, values: &[R::E], omega: &R::E) -> Vec<R::E>
    where
        R: Ring,
        R::E: RingOps,
    {
        (0..values.len())
            .map(|k| {
                let mut acc = ring.zero();
                for (j, v) in values.iter().enumerate() {
                    acc = acc + v.clone() * pow(ring, omega, j * k);
                }
                acc
            })
            .collect()
    }

    #[test]
    fn agrees_with_the_naive_transform_and_inverts() {
        // 2 has order 8 in F_17: 2^4 = 16, 2^8 = 1.
        let f17 = Integers::modulo(17);
        let omega = f17.element(2);
        assert_eq!(pow(&f17, &omega, 8), f17.identity());
        assert_ne!(pow(&f17, &omega, 4), f17.identity());

        let original: Vec<_> = (1..=8).map(|n| f17.element(n)).collect();

        let mut transformed = original.clone();
        fft(&f17, &mut transformed, &omega);
        assert_eq!(transformed, naive_dft(&f17, &original, &omega));

        let omega_inv = f17.invert(&omega).expect("omega is a unit");
        let n_inv = f17.invert(&f17.from_integer(8)).expect("8 is a unit mod 17");
        inverse_fft(&f17, &mut transformed, &omega_inv, &n_inv);
        assert_eq!(transformed, original);
    }
}
