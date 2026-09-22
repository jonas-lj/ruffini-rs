//! Matrices over a ring, and the ring of square matrices over one.
//!
//! [`Matrix`] is any shape and carries its base ring, so products of compatible
//! rectangles work. [`MatrixRing`] is the `n × n` matrices over a ring, which is what
//! forms a [`Ring`] itself.

use crate::structures::{
    AdditiveGroup, CommutativeMonoid, Domain, Field, Monoid, Ring, RingOps, SemiRing, Semigroup,
};
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
}

impl<R> Matrix<R>
where
    R: Ring + Clone,
    R::E: RingOps + Eq,
{
    /// The determinant, by cofactor expansion along the first column.
    ///
    /// Needs no division, so it works over any ring, but costs `O(n!)`; over a field
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
    /// Gauss-Jordan elimination on `[self | I]`; needs a field to scale pivot rows.
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
    fn addition_and_multiplication() {
        let r = zm(2);
        let a = r.element(vec![int(1), int(2), int(3), int(4)]);
        let b = r.element(vec![int(0), int(1), int(1), int(0)]);

        // Entry-wise addition.
        assert_eq!(
            a.clone() + b.clone(),
            r.element(vec![int(1), int(3), int(4), int(4)])
        );

        // b swaps the columns of a.
        assert_eq!(
            a.clone() * b.clone(),
            r.element(vec![int(2), int(1), int(4), int(3)])
        );
        // ...and swaps a's rows on the other side, so it does not commute.
        assert_eq!(
            b.clone() * a.clone(),
            r.element(vec![int(3), int(4), int(1), int(2)])
        );
        assert_ne!(a.clone() * b.clone(), b * a.clone());

        // Ring axioms.
        assert_eq!(a.clone() * r.identity(), a);
        assert_eq!(a.clone() + r.zero(), a);
        assert_eq!(a.clone() - a.clone(), r.zero());

        // Rectangles multiply when the inner dimensions agree: 2x3 by 3x2.
        let p = m(2, 3, &[1, 2, 3, 4, 5, 6]);
        let q = m(3, 2, &[7, 8, 9, 10, 11, 12]);
        assert_eq!(p.clone() * q.clone(), m(2, 2, &[58, 64, 139, 154]));
        assert_eq!(q * p, m(3, 3, &[39, 54, 69, 49, 68, 87, 59, 82, 105]));
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
        assert_eq!(
            (a.clone() * b.clone()).determinant(),
            a.determinant() * b.determinant()
        );
    }

    #[test]
    fn inversion_over_a_field() {
        // Needs a field to scale pivot rows, so work in F_7.
        let f7 = Integers::modulo(7);
        let r = f7.matrices(3);
        let e = |n: i64| f7.element(n);

        let a = r.element(vec![e(2), e(1), e(0), e(1), e(1), e(0), e(0), e(0), e(3)]);
        let inv = a.inverse().expect("invertible over F_7");
        assert_eq!(a.clone() * inv.clone(), r.identity());
        assert_eq!(inv * a, r.identity());

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
            e(4) * det_inv.clone(),
            (e(0) - e(2)) * det_inv.clone(),
            (e(0) - e(3)) * det_inv.clone(),
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
