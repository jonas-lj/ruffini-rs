//! Matrices over a ring, and the ring of square matrices over one.
//!
//! [`Matrix`] is any shape and carries its base ring, so products of compatible
//! rectangles work. [`MatrixRing`] is the `n × n` matrices over a ring, which is what
//! forms a [`Ring`] itself.

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, Domain, Field, Monoid, Ring, RingOps, SemiRing, Semigroup,
};
use num_bigint::BigInt;
use std::fmt;
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub};
use std::rc::Rc;

/// A matrix over `R`, stored dense in row-major order.
#[derive(Debug)]
pub struct Matrix<R: Ring + Clone>
where
    R::E: RingOps,
{
    rows: usize,
    cols: usize,
    entries: Vec<R::E>,
    ring: R,
}

/// The ring of `n × n` matrices over `R`.
#[derive(Debug)]
pub struct MatrixRing<R: Ring + Clone>
where
    R::E: RingOps,
{
    base: R,
    dimension: usize,
}

impl<R> Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps,
{
    /// Builds a `rows × cols` matrix from entries in row-major order.
    ///
    /// # Panics
    /// If the entry count is not `rows * cols`.
    pub fn new(ring: R, rows: usize, cols: usize, entries: Vec<R::E>) -> Self {
        assert_eq!(
            entries.len(),
            rows * cols,
            "expected {rows} * {cols} entries"
        );
        Matrix {
            rows,
            cols,
            entries,
            ring,
        }
    }

    /// Builds a matrix from a function of the row and column index.
    pub fn from_fn(ring: R, rows: usize, cols: usize, mut f: impl FnMut(usize, usize) -> R::E) -> Self {
        let entries = (0..rows).flat_map(|i| (0..cols).map(move |j| (i, j)));
        let entries = entries.map(|(i, j)| f(i, j)).collect();
        Matrix::new(ring, rows, cols, entries)
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// The base ring the entries live in.
    pub fn base(&self) -> &R {
        &self.ring
    }

    /// The entry at row `i`, column `j`.
    pub fn get(&self, i: usize, j: usize) -> &R::E {
        &self.entries[i * self.cols + j]
    }

    /// A copy with row `i` and column `j` removed.
    pub fn minor(&self, i: usize, j: usize) -> Self {
        let entries = (0..self.rows)
            .filter(|&r| r != i)
            .flat_map(|r| {
                (0..self.cols)
                    .filter(|&c| c != j)
                    .map(move |c| (r, c))
            })
            .map(|(r, c)| self.get(r, c).clone())
            .collect();
        Matrix::new(self.ring.clone(), self.rows - 1, self.cols - 1, entries)
    }

    /// The transpose.
    pub fn transpose(&self) -> Self {
        Matrix::from_fn(self.ring.clone(), self.cols, self.rows, |i, j| {
            self.get(j, i).clone()
        })
    }

    /// Every entry multiplied by `c` on the left.
    ///
    /// For a square matrix this is `c·I * self`, at `rows * cols` multiplications
    /// rather than `n^3`; unlike that product it is also defined for a rectangle.
    /// The side matters: a coefficient ring need not be commutative.
    ///
    /// A coefficient cannot go on either side of `*`, because nothing rules out `R::E`
    /// being `Matrix<R>` and the impl is then judged to overlap with matrix
    /// multiplication. Plain integers can, and do.
    pub fn scale(&self, c: &R::E) -> Self {
        Matrix::from_fn(self.ring.clone(), self.rows, self.cols, |i, j| {
            c.clone() * self.get(i, j).clone()
        })
    }
}

impl<R> Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
    /// The determinant, by cofactor expansion along the first column.
    ///
    /// Needs no division, so it works over any ring, but costs `O(n!)`. Over a field
    /// Gaussian elimination would be `O(n^3)`.
    ///
    /// # Panics
    /// If the matrix is not square, or is empty.
    pub fn determinant(&self) -> R::E {
        assert!(self.is_square(), "determinant of a non-square matrix");
        assert!(self.rows > 0, "determinant of an empty matrix");
        if self.rows == 1 {
            return self.get(0, 0).clone();
        }
        let mut total = self.ring.zero();
        let mut sign = self.ring.identity();
        for i in 0..self.rows {
            let term = self.get(i, 0).clone() * sign.clone() * self.minor(i, 0).determinant();
            total += term;
            sign = self.ring.zero() - sign;
        }
        total
    }
}

impl<R> Matrix<R>
where
    R: Field + Clone,
    R::E: RingOps + Eq,
{
    /// The inverse, or [`None`] if the matrix is singular.
    ///
    /// Gauss-Jordan elimination on `[self | I]`, which needs a field to scale pivot rows.
    ///
    /// # Panics
    /// If the matrix is not square.
    pub fn inverse(&self) -> Option<Self> {
        assert!(self.is_square(), "inverse of a non-square matrix");
        let n = self.rows;
        let (zero, one) = (self.ring.zero(), self.ring.identity());

        // Work on [self | I], row-reduce the left half to I.
        let mut a: Vec<Vec<R::E>> = (0..n)
            .map(|i| {
                (0..2 * n)
                    .map(|j| {
                        if j < n {
                            self.get(i, j).clone()
                        } else if j - n == i {
                            one.clone()
                        } else {
                            zero.clone()
                        }
                    })
                    .collect()
            })
            .collect();

        for col in 0..n {
            let pivot = (col..n).find(|&r| a[r][col] != zero)?;
            a.swap(col, pivot);

            let scale = self.ring.invert(&a[col][col])?;
            for entry in a[col].iter_mut().skip(col) {
                *entry = entry.clone() * scale.clone();
            }

            for r in 0..n {
                if r == col || a[r][col] == zero {
                    continue;
                }
                let factor = a[r][col].clone();
                let pivot_row: Vec<R::E> = a[col][col..].to_vec();
                for (entry, p) in a[r].iter_mut().skip(col).zip(pivot_row) {
                    let subtract = factor.clone() * p;
                    *entry = entry.clone() - subtract;
                }
            }
        }

        let entries = a.into_iter().flat_map(|row| row.into_iter().skip(n)).collect();
        Some(Matrix::new(self.ring.clone(), n, n, entries))
    }
}

impl<R> MatrixRing<R>
where
    R: Ring + Clone,
    R::E: RingOps,
{
    /// The ring of `n × n` matrices over `base`.
    pub fn new(base: R, dimension: usize) -> Rc<Self> {
        Rc::new(MatrixRing { base, dimension })
    }

    /// The ring the entries live in.
    pub fn base(&self) -> &R {
        &self.base
    }

    /// The side length of these matrices.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

impl<R> Clone for Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps,
{
    fn clone(&self) -> Self {
        Matrix {
            rows: self.rows,
            cols: self.cols,
            entries: self.entries.clone(),
            ring: self.ring.clone(),
        }
    }
}

impl<R> PartialEq for Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.entries == other.entries
    }
}

impl<R> Eq for Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
}

/// Entry-wise, for matrices of the same shape.
macro_rules! matrix_entrywise_ops {
    ($($op:ident, $method:ident);* $(;)?) => {$(
        impl<R> $op for Matrix<R>
        where
            R: Ring + Clone,
            R::E: RingOps,
        {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self::Output {
                assert_eq!(
                    (self.rows, self.cols),
                    (rhs.rows, rhs.cols),
                    "matrix shapes do not match"
                );
                let entries = self
                    .entries
                    .into_iter()
                    .zip(rhs.entries)
                    .map(|(x, y)| x.$method(y))
                    .collect();
                Matrix::new(self.ring, self.rows, self.cols, entries)
            }
        }
    )*};
}
matrix_entrywise_ops!(Add, add; Sub, sub);

/// Row-by-column, for `m × k` times `k × n`.
impl<R> Mul for Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps,
{
    type Output = Self;
    fn mul(self, rhs: Self) -> Self::Output {
        assert_eq!(
            self.cols, rhs.rows,
            "cannot multiply {}x{} by {}x{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        let ring = self.ring.clone();
        Matrix::from_fn(ring, self.rows, rhs.cols, |i, j| {
            (0..self.cols).fold(self.ring.zero(), |sum, k| {
                sum + self.get(i, k).clone() * rhs.get(k, j).clone()
            })
        })
    }
}

/// Entry-wise on borrowed operands, reading both through the entries' reference ops.
macro_rules! matrix_borrowed_entrywise_ops {
    ($($op:ident, $method:ident);* $(;)?) => {$(
        impl<R> $op<&Matrix<R>> for &Matrix<R>
        where
            R: Ring + Clone,
            R::E: RingOps,
            for<'c> &'c R::E: $op<&'c R::E, Output = R::E>,
        {
            type Output = Matrix<R>;
            fn $method(self, rhs: &Matrix<R>) -> Self::Output {
                assert_eq!(
                    (self.rows, self.cols),
                    (rhs.rows, rhs.cols),
                    "matrix shapes do not match"
                );
                let entries = self
                    .entries
                    .iter()
                    .zip(&rhs.entries)
                    .map(|(x, y)| x.$method(y))
                    .collect();
                Matrix::new(self.ring.clone(), self.rows, self.cols, entries)
            }
        }
    )*};
}
matrix_borrowed_entrywise_ops!(Add, add; Sub, sub);

impl<R> Mul<&Matrix<R>> for &Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps,
    for<'c> &'c R::E: Mul<&'c R::E, Output = R::E>,
{
    type Output = Matrix<R>;
    fn mul(self, rhs: &Matrix<R>) -> Self::Output {
        assert_eq!(
            self.cols, rhs.rows,
            "cannot multiply {}x{} by {}x{}",
            self.rows, self.cols, rhs.rows, rhs.cols
        );
        Matrix::from_fn(self.ring.clone(), self.rows, rhs.cols, |i, j| {
            (0..self.cols).fold(self.ring.zero(), |sum, k| {
                sum + self.get(i, k) * rhs.get(k, j)
            })
        })
    }
}

forward_ref_binops!(
    Matrix<R>,
    { R: Ring + Clone, R::E: RingOps, },
    Add, add; Sub, sub; Mul, mul
);

/// Scaling by a plain integer, embedded in the coefficient ring first.
///
/// `Mul` only. `m * 2` is unambiguous, since scaling entrywise and multiplying by
/// `2·I` agree wherever both are defined, and scaling is the one that also works on a
/// rectangle. `m + 1` would not be: adding the identity matrix and adding one to every
/// entry are different matrices.
///
/// A generic coefficient cannot have these impls. Nothing rules out `R::E` being
/// `Matrix<R>`, so `impl<R> Mul<Matrix<R>> for R::E` is judged to overlap with matrix
/// multiplication, and so is the same impl with the operands the other way round. A
/// concrete integer type is never a `Matrix`, so it is free of that. Use
/// [`Matrix::scale`] for a coefficient.
macro_rules! matrix_scalar_mul {
    ($($int:ty),+ $(,)?) => {$(
        impl<R> Mul<$int> for Matrix<R>
        where
            R: Ring + Clone,
            R::E: RingOps,
        {
            type Output = Matrix<R>;
            fn mul(self, rhs: $int) -> Matrix<R> {
                let c = self.ring.from_integer(rhs);
                self.scale(&c)
            }
        }

        impl<R> Mul<$int> for &Matrix<R>
        where
            R: Ring + Clone,
            R::E: RingOps,
        {
            type Output = Matrix<R>;
            fn mul(self, rhs: $int) -> Matrix<R> {
                let c = self.ring.from_integer(rhs);
                self.scale(&c)
            }
        }

        impl<R> Mul<Matrix<R>> for $int
        where
            R: Ring + Clone,
            R::E: RingOps,
        {
            type Output = Matrix<R>;
            fn mul(self, rhs: Matrix<R>) -> Matrix<R> {
                let c = rhs.ring.from_integer(self);
                rhs.scale(&c)
            }
        }

        impl<R> Mul<&Matrix<R>> for $int
        where
            R: Ring + Clone,
            R::E: RingOps,
        {
            type Output = Matrix<R>;
            fn mul(self, rhs: &Matrix<R>) -> Matrix<R> {
                let c = rhs.ring.from_integer(self);
                rhs.scale(&c)
            }
        }
    )+};
}
matrix_scalar_mul!(i64, BigInt);

/// Compound assignment, written directly so it carries no binder.
macro_rules! matrix_assign_ops {
    ($($op:ident, $method:ident, $base_method:ident);* $(;)?) => {$(
        impl<R> $op for Matrix<R>
        where
            R: Ring + Clone,
            R::E: RingOps,
        {
            fn $method(&mut self, rhs: Self) {
                let placeholder = Matrix {
                    rows: 0,
                    cols: 0,
                    entries: Vec::new(),
                    ring: self.ring.clone(),
                };
                let lhs = std::mem::replace(self, placeholder);
                *self = lhs.$base_method(rhs);
            }
        }
    )*};
}
matrix_assign_ops!(AddAssign, add_assign, add; MulAssign, mul_assign, mul);

impl<R> Domain for Rc<MatrixRing<R>>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
    type E = Matrix<R>;
    type Repr = Vec<R::E>;

    /// Entries in row-major order.
    ///
    /// # Panics
    /// If the entry count is not `dimension * dimension`.
    fn element<T: Into<Self::Repr>>(&self, value: T) -> Self::E {
        Matrix::new(
            self.base.clone(),
            self.dimension,
            self.dimension,
            value.into(),
        )
    }
}

impl<R> Semigroup for Rc<MatrixRing<R>>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
}

impl<R> Monoid for Rc<MatrixRing<R>>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
    fn identity(&self) -> Self::E {
        let (zero, one) = (self.base.zero(), self.base.identity());
        Matrix::from_fn(self.base.clone(), self.dimension, self.dimension, |i, j| {
            if i == j {
                one.clone()
            } else {
                zero.clone()
            }
        })
    }
}

impl<R> CommutativeMonoid for Rc<MatrixRing<R>>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
    fn zero(&self) -> Self::E {
        let zero = self.base.zero();
        Matrix::from_fn(self.base.clone(), self.dimension, self.dimension, |_, _| {
            zero.clone()
        })
    }
}

impl<R> AdditiveGroup for Rc<MatrixRing<R>>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
}

impl<R> SemiRing for Rc<MatrixRing<R>>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
}

impl<R> Ring for Rc<MatrixRing<R>>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
}

impl<R> fmt::Display for Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for i in 0..self.rows {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "[")?;
            for j in 0..self.cols {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.get(i, j))?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
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

    fn zm(n: usize) -> Rc<MatrixRing<Integers>> {
        Integers::default().matrices(n)
    }

    /// Row-major, over Z.
    fn m(rows: usize, cols: usize, e: &[i64]) -> Matrix<Integers> {
        Matrix::new(
            Integers::default(),
            rows,
            cols,
            e.iter().map(|n| int(*n)).collect(),
        )
    }

    #[test]
    // `0 * m` is the case under test, not a slip.
    #[allow(clippy::erasing_op)]
    fn scaling_by_an_integer_agrees_with_multiplying_by_n_times_the_identity() {
        let rectangle = m(2, 3, &[1, 2, 3, 4, 5, 6]);
        let expected = m(2, 3, &[2, 4, 6, 8, 10, 12]);

        // All four operand combinations, on a shape that has no identity matrix.
        assert_eq!(2 * rectangle.clone(), expected);
        assert_eq!(2 * &rectangle, expected);
        assert_eq!(rectangle.clone() * 2, expected);
        assert_eq!(&rectangle * 2, expected);
        assert_eq!(rectangle.scale(&int(2)), expected);
        assert_eq!(BigInt::from(2) * rectangle.clone(), expected);
        assert_eq!(rectangle.clone() * BigInt::from(2), expected);

        // Where both are defined they are the same matrix, which is what makes the
        // operator unambiguous.
        let square = m(2, 2, &[1, 2, 3, 4]);
        let two_i = m(2, 2, &[2, 0, 0, 2]);
        assert_eq!(2 * square.clone(), &two_i * &square);

        // Zero and one are not special-cased.
        assert_eq!(0 * square.clone(), m(2, 2, &[0, 0, 0, 0]));
        assert_eq!(1 * square.clone(), square);
        assert_eq!(-1 * square.clone(), m(2, 2, &[-1, -2, -3, -4]));
    }

    #[test]
    fn scale_multiplies_the_coefficient_on_the_left() {
        // Entries that do not commute: 2x2 matrices over Z.
        let z2 = zm(2);
        let a = m(2, 2, &[0, 1, 0, 0]);
        let b = m(2, 2, &[0, 0, 1, 0]);
        assert_ne!(&a * &b, &b * &a);

        let one_by_one = Matrix::new(z2.clone(), 1, 1, vec![b.clone()]);
        assert_eq!(one_by_one.scale(&a).get(0, 0), &(&a * &b));
        assert_ne!(one_by_one.scale(&a).get(0, 0), &(&b * &a));
    }

    #[test]
    fn addition_and_multiplication() {
        let r = zm(2);
        let a = r.element(vec![int(1), int(2), int(3), int(4)]);
        let b = r.element(vec![int(0), int(1), int(1), int(0)]);

        // Entry-wise addition.
        assert_eq!(&a + &b, r.element(vec![int(1), int(3), int(4), int(4)]));

        // b swaps the columns of a.
        assert_eq!(&a * &b, r.element(vec![int(2), int(1), int(4), int(3)]));
        // ...and swaps a's rows on the other side, so it does not commute.
        assert_eq!(&b * &a, r.element(vec![int(3), int(4), int(1), int(2)]));
        assert_ne!(&a * &b, &b * &a);
        // All four owned/borrowed combinations resolve.
        assert_eq!(a.clone() * &b, &a * b.clone());

        // Ring axioms.
        assert_eq!(&a * r.identity(), a);
        assert_eq!(&a + r.zero(), a);
        assert_eq!(&a - &a, r.zero());

        // Rectangles multiply when the inner dimensions agree: 2x3 by 3x2.
        let p = m(2, 3, &[1, 2, 3, 4, 5, 6]);
        let q = m(3, 2, &[7, 8, 9, 10, 11, 12]);
        assert_eq!(&p * &q, m(2, 2, &[58, 64, 139, 154]));
        assert_eq!(&q * &p, m(3, 3, &[39, 54, 69, 49, 68, 87, 59, 82, 105]));
    }

    #[test]
    #[should_panic(expected = "cannot multiply")]
    fn rejects_incompatible_shapes() {
        let _ = m(2, 3, &[1, 2, 3, 4, 5, 6]) * m(2, 3, &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn determinant_by_cofactor_expansion() {
        assert_eq!(m(1, 1, &[7]).determinant(), int(7));
        // 1*4 - 2*3
        assert_eq!(m(2, 2, &[1, 2, 3, 4]).determinant(), int(-2));
        // Singular: second row is twice the first.
        assert_eq!(m(2, 2, &[1, 2, 2, 4]).determinant(), int(0));
        assert_eq!(m(3, 3, &[6, 1, 1, 4, -2, 5, 2, 8, 7]).determinant(), int(-306));
        // Triangular: the product of the diagonal.
        assert_eq!(m(3, 3, &[2, 9, 9, 0, 3, 9, 0, 0, 5]).determinant(), int(30));

        let r = zm(3);
        assert_eq!(r.identity().determinant(), int(1));
        assert_eq!(r.zero().determinant(), int(0));

        // det(AB) = det(A) det(B).
        let a = m(3, 3, &[1, 2, 0, 3, -1, 2, 0, 4, 1]);
        let b = m(3, 3, &[2, 0, 1, 1, 3, 0, 0, 2, 4]);
        assert_eq!((&a * &b).determinant(), a.determinant() * b.determinant());
    }

    #[test]
    fn inversion_over_a_field() {
        // Needs a field to scale pivot rows, so work in F_7.
        let f7 = Integers::modulo(7);
        let r = f7.matrices(3);
        let e = |n: i64| f7.element(n);

        let a = r.element(vec![e(2), e(1), e(0), e(1), e(1), e(0), e(0), e(0), e(3)]);
        let inv = a.inverse().expect("invertible over F_7");
        assert_eq!(&a * &inv, r.identity());
        assert_eq!(&inv * &a, r.identity());

        // The identity inverts to itself.
        assert_eq!(r.identity().inverse().unwrap(), r.identity());

        // Singular: the second row is the first, so there is no inverse.
        let singular = r.element(vec![e(1), e(2), e(3), e(1), e(2), e(3), e(0), e(0), e(1)]);
        assert_eq!(singular.inverse(), None);
        assert_eq!(r.zero().inverse(), None);

        // Inverting a 2x2 and checking against the closed form.
        let r2 = f7.matrices(2);
        let b = r2.element(vec![e(1), e(2), e(3), e(4)]);
        let det = b.determinant(); // 4 - 6 = -2 = 5 mod 7
        assert_eq!(det, e(5));
        let det_inv = f7.invert(&det).unwrap();
        let expected = r2.element(vec![
            e(4) * &det_inv,
            (e(0) - e(2)) * &det_inv,
            (e(0) - e(3)) * &det_inv,
            e(1) * det_inv,
        ]);
        assert_eq!(b.inverse().unwrap(), expected);
    }

    #[test]
    fn shape_helpers_and_display() {
        let p = m(2, 3, &[1, 2, 3, 4, 5, 6]);
        assert_eq!((p.rows(), p.cols()), (2, 3));
        assert!(!p.is_square());
        assert_eq!(p.get(1, 2), &int(6));
        assert_eq!(p.transpose(), m(3, 2, &[1, 4, 2, 5, 3, 6]));
        assert_eq!(p.transpose().transpose(), p);
        // Dropping row 0 and column 1 leaves [[4, 6]].
        assert_eq!(p.minor(0, 1), m(1, 2, &[4, 6]));
        assert_eq!(format!("{}", m(2, 2, &[1, 2, 3, 4])), "[[1, 2], [3, 4]]");
    }
}
