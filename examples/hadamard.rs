//! Williamson's construction of a Hadamard matrix of order 4n.
//!
//! Follows J. S. Wallis, "Construction of Williamson type matrices", Linear and
//! Multilinear Algebra 3 (1975) 197-207.
//!
//! Run with `cargo run --release --example hadamard -- [n]`, default n = 11. The
//! method needs `(n-1)/2 >= 4`, so n = 9 is the smallest it can do. n = 23 gives
//! order 92, which is the order the Wallis paper set out to construct, and takes
//! about half a minute.

use num_bigint::BigInt;
use ruffini::integers::{Integer, Integers};
use ruffini::matrices::{Matrix, MatrixRing};
use ruffini::polynomials::{Polynomial, PolynomialRing, RingExt};
use ruffini::structures::{CommutativeMonoid, Domain, Monoid, Ring};
use std::rc::Rc;
use std::time::Instant;

type Zx = Rc<PolynomialRing<Integers>>;
type Zp = Polynomial<Integers>;

/// The group ring `Z[x]/(x^n - 1)`: polynomials reduced modulo a monic divisor, which
/// needs no field.
struct GroupRing {
    ring: Zx,
    modulus: Zp,
}

impl GroupRing {
    fn new(n: usize) -> Self {
        let ring = Integers::default().polynomials();
        let x = ring.indeterminate();
        let modulus = x.pow(n) - 1;
        GroupRing { ring, modulus }
    }

    fn reduce(&self, p: Zp) -> Zp {
        p.div_rem_monic(&self.modulus).1
    }

    fn multiply(&self, a: &Zp, b: &Zp) -> Zp {
        self.reduce(a * b)
    }

    /// `2x^j + 2x^(n-j)`, the symmetric generators Williamson's method is built from.
    ///
    /// The coefficient is 2, not 1: it is what makes `(sum B)/2 - B_i` land on +/-1
    /// at every position later on.
    fn v(&self, n: usize, j: usize) -> Zp {
        let x = self.ring.indeterminate();
        2 * x.pow(j) + 2 * x.pow(n - j)
    }
}

/// `m` with `k^2 = (1 + 4m)^2`, defined for odd `k`.
fn compute_m(k: i64) -> i64 {
    match k.rem_euclid(4) {
        1 => (k - 1) / 4,
        3 => -((k + 1) / 4),
        _ => panic!("k must be odd"),
    }
}

/// Every ordered partition of `0..size` into `parts` non-empty blocks.
fn partitions(size: usize, parts: usize) -> Vec<Vec<Vec<usize>>> {
    let mut out = Vec::new();
    // Assign each index a block, then keep the assignments that leave none empty.
    let mut assignment = vec![0usize; size];
    loop {
        let mut blocks = vec![Vec::new(); parts];
        for (i, &b) in assignment.iter().enumerate() {
            blocks[b].push(i);
        }
        if blocks.iter().all(|b| !b.is_empty()) {
            out.push(blocks);
        }
        // Odometer over base `parts`.
        let mut i = 0;
        loop {
            if i == size {
                return out;
            }
            assignment[i] += 1;
            if assignment[i] < parts {
                break;
            }
            assignment[i] = 0;
            i += 1;
        }
    }
}

/// Every split of `set` into `(positive, negative)` with `#positive - #negative == d`.
fn splits(set: &[usize], d: i64) -> Vec<(Vec<usize>, Vec<usize>)> {
    let len = set.len() as i64;
    if len < d.abs() || (len - d) % 2 != 0 {
        return Vec::new();
    }
    let take = ((len + d) / 2) as usize;
    let mut out = Vec::new();
    // Enumerate subsets of the right size by bitmask.
    for mask in 0u32..(1 << set.len()) {
        if mask.count_ones() as usize != take {
            continue;
        }
        let mut positive = Vec::new();
        let mut negative = Vec::new();
        for (i, &x) in set.iter().enumerate() {
            if mask >> i & 1 == 1 {
                positive.push(x);
            } else {
                negative.push(x);
            }
        }
        out.push((positive, negative));
    }
    out
}

/// `H * H^T == 4n * I`, the defining property.
fn is_hadamard(h: &Matrix<Integers>, order: usize) -> bool {
    let expected = order as i64 * Integers::default().matrices(order).identity();
    h * &h.transpose() == expected
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(11);
    let order = 4 * n;
    println!("Searching for a Hadamard matrix of order {order} = 4 x {n}");

    let decompositions = four_odd_squares(order);
    assert!(!decompositions.is_empty(), "{order} is not a sum of four odd squares");

    let group = GroupRing::new(n);
    let half = (n - 1) / 2;
    let generators: Vec<Zp> = (1..=half).map(|j| group.v(n, j)).collect();
    let target = group.ring.from_integer(order as i64);

    // Which decomposition is used matters: one can fail where another succeeds, so
    // try each in turn.
    let start = Instant::now();
    let mut found = None;
    for k in &decompositions {
        let m: Vec<i64> = k.iter().map(|&ki| compute_m(ki)).collect();
        println!("  trying 4n = {k:?} squared and summed, giving m = {m:?}");
        found = search(&group, half, &generators, &target, &m);
        if found.is_some() {
            break;
        }
    }

    let Some(b) = found else {
        println!("  no Williamson set found for n = {n}");
        return;
    };
    println!("  found a Williamson set in {:?}", start.elapsed());
    finish(&group, &b, n, order);
}

/// Searches every partition and split for a type III set.
fn search(
    group: &GroupRing,
    half: usize,
    generators: &[Zp],
    target: &Zp,
    m: &[i64],
) -> Option<Vec<Zp>> {
    for blocks in partitions(half, 4) {
        // Each block must be splittable with the right difference.
        if (0..4).any(|i| {
            let len = blocks[i].len() as i64;
            len < m[i].abs() || (len - m[i]) % 2 != 0
        }) {
            continue;
        }
        let per_block: Vec<_> = (0..4).map(|i| splits(&blocks[i], m[i])).collect();

        for s0 in &per_block[0] {
            for s1 in &per_block[1] {
                for s2 in &per_block[2] {
                    for s3 in &per_block[3] {
                        let b: Vec<Zp> = [s0, s1, s2, s3]
                            .iter()
                            .map(|(positive, negative)| {
                                let mut acc = group.ring.identity();
                                for &j in positive.iter() {
                                    acc += generators[j].clone();
                                }
                                for &j in negative.iter() {
                                    acc = acc - generators[j].clone();
                                }
                                group.reduce(acc)
                            })
                            .collect();

                        // Type III: the squares sum to the constant 4n.
                        let sum_of_squares = b.iter().fold(group.ring.zero(), |acc, bi| {
                            acc + group.multiply(bi, bi)
                        });
                        if &sum_of_squares == target {
                            return Some(b);
                        }
                    }
                }
            }
        }
    }
    None
}

/// Builds the matrix from a type III set and checks it.
fn finish(group: &GroupRing, b: &[Zp], n: usize, order: usize) {
    // Type III to type I: A_i = (sum B)/2 - B_i.
    let total = b.iter().fold(group.ring.zero(), |acc, bi| acc + bi.clone());
    let halved = group.ring.element(
        total
            .coefficients()
            .iter()
            .map(halve)
            .collect::<Vec<_>>(),
    );
    let a: Vec<Zp> = b.iter().map(|bi| halved.clone() - bi.clone()).collect();

    // Apply each A_i to the cyclic shift matrix, giving n x n circulants.
    let matrices = Integers::default().matrices(n);
    let u = Matrix::from_fn(Integers::default(), n, n, |i, j| {
        Integer::from(if i == (j + n - 1) % n { 1 } else { 0 })
    });
    let blocks: Vec<Matrix<Integers>> = a.iter().map(|ai| apply(ai, &u, &matrices)).collect();

    let h = williamson_array(&blocks, n);
    println!("  built a {order} x {order} matrix");
    println!("  H * H^T == {order} I  ->  {}", is_hadamard(&h, order));
    println!("  entries are all +/-1  ->  {}", all_plus_minus_one(&h));
}

/// Evaluates a polynomial with integer coefficients at a matrix, by Horner.
fn apply(p: &Zp, u: &Matrix<Integers>, ring: &Rc<MatrixRing<Integers>>) -> Matrix<Integers> {
    // The constant term of each step is `c * I`, which `scale` reaches directly -
    // an operator cannot, since the coefficient is a ring element.
    p.coefficients().iter().rev().fold(ring.zero(), |acc, c| {
        acc * u.clone() + ring.identity().scale(c)
    })
}

/// The Williamson array: a 4x4 block matrix in the four circulants.
fn williamson_array(b: &[Matrix<Integers>], n: usize) -> Matrix<Integers> {
    let neg = |m: &Matrix<Integers>| -1 * m;
    let layout = [
        [b[0].clone(), b[1].clone(), b[2].clone(), b[3].clone()],
        [neg(&b[1]), b[0].clone(), neg(&b[3]), b[2].clone()],
        [neg(&b[2]), b[3].clone(), b[0].clone(), neg(&b[1])],
        [neg(&b[3]), neg(&b[2]), b[1].clone(), b[0].clone()],
    ];
    Matrix::from_fn(Integers::default(), 4 * n, 4 * n, |i, j| {
        layout[i / n][j / n].get(i % n, j % n).clone()
    })
}

/// `c / 2`, exact for the coefficients this is used on.
fn halve(c: &Integer) -> Integer {
    let value: BigInt = c.clone().into();
    Integer::from(value / 2)
}

/// Every way to write `total` as four odd squares, largest first, up to order.
///
/// Which one is used matters: a decomposition can fail to admit a Williamson set
/// while another for the same `n` succeeds.
fn four_odd_squares(total: usize) -> Vec<Vec<i64>> {
    let total = total as i64;
    let odds: Vec<i64> = (1..).map(|i| 2 * i - 1).take_while(|o| o * o <= total).collect();
    let mut out = Vec::new();
    for &a in odds.iter().rev() {
        for &b in odds.iter().rev().filter(|&&b| b <= a) {
            for &c in odds.iter().rev().filter(|&&c| c <= b) {
                for &d in odds.iter().rev().filter(|&&d| d <= c) {
                    if a * a + b * b + c * c + d * d == total {
                        out.push(vec![a, b, c, d]);
                    }
                }
            }
        }
    }
    out
}

fn all_plus_minus_one(h: &Matrix<Integers>) -> bool {
    let (one, minus_one) = (Integer::from(1), Integer::from(-1));
    (0..h.rows()).all(|i| {
        (0..h.cols()).all(|j| {
            let e = h.get(i, j);
            *e == one || *e == minus_one
        })
    })
}
