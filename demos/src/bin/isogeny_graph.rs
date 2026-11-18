//! The supersingular 2-isogeny graph over `F_{p^2}`.
//!
//! Every supersingular curve in characteristic `p` can be written over `F_{p^2}`, and
//! there are only finitely many up to isomorphism: Deuring counted `floor(p/12) + e`,
//! with `e` of `0, 1, 1, 2` as `p` is `1, 5, 7, 11` modulo `12`. Take those classes as
//! vertices, named by their j-invariants, and draw an edge wherever a 2-isogeny goes.
//! The graph that comes out is connected and 3-regular, and walking it is the hard
//! problem behind isogeny-based cryptography: finding a path between two given
//! vertices.
//!
//! This walks all of it for a small `p`, in these steps.
//!
//! 1. Build `F_{p^2}`, which is `F_p[x]` modulo an irreducible quadratic.
//! 2. Start from `y^2 = x^3 + x`, which is supersingular exactly when `p = 3 (mod 4)`.
//! 3. Search breadth-first. A point of order two on `y^2 = x^3 + ax + b` is an
//!    `(x, 0)` with `x` a root of the cubic, so the cubic's roots are the three
//!    kernels of a 2-isogeny leaving a curve. Vélu's formulas turn each kernel into a
//!    codomain, whose j-invariant says which vertex has been reached.
//! 4. Report the adjacency.
//! 5. Check five things against the theory.
//!
//! Everything here is brute force: the roots are found by trying every element of the
//! field, and the vertices are a `Vec` searched linearly. Both are fine at the sizes
//! where the graph still fits on a page.
//!
//! Silverman, "The Arithmetic of Elliptic Curves", V.4 for the count, III.6 for the
//! dual isogeny. Run with `cargo run --release --bin isogeny_graph -- [p]`, where `p`
//! is a prime congruent to 3 modulo 4 and defaults to 23. The cost is nearly all in
//! counting points to confirm supersingularity, `p^2` characters per curve: a second
//! for `p = 47`, a quarter of a minute for `p = 83`.

use num_bigint::BigInt;
use ruffini::elliptic::Curve;
use ruffini::finite_fields::{finite_field, ExtensionField, FiniteField, PrimeField};
use ruffini::polynomials::{Polynomial, PolynomialRing};
use ruffini::structures::{CommutativeMonoid, QuotientRingElement, Ring};
use std::sync::Arc;

type Field = ExtensionField;
type Element = QuotientRingElement<Arc<PolynomialRing<PrimeField>>>;

/// A vertex: an isomorphism class of supersingular curves, named by its j-invariant,
/// holding whichever curve of the class the walk arrived on.
struct Vertex {
    j_invariant: Element,
    curve: Arc<Curve<Field>>,
}

fn main() {
    let p: u32 = std::env::args()
        .nth(1)
        .map_or(23, |argument| argument.parse().expect("p must be a number"));
    assert_eq!(
        p % 4,
        3,
        "y^2 = x^3 + x is supersingular only for p = 3 (mod 4)"
    );
    println!("The supersingular 2-isogeny graph over F_{p}^2\n");

    // Step 1. F_{p^2} is F_p[x] modulo an irreducible quadratic, which the library
    // looks up or searches for.
    let field = finite_field(p, 2);
    println!("1. The field");
    println!(
        "   F_{p}^2 = F_{p}[x]/({}), of order {}",
        show_polynomial(p, field.modulus()),
        field.order()
    );

    // Step 2. The starting curve, whose j-invariant is 1728 reduced modulo p.
    let start = Curve::new(field.clone(), field.from_integer(1), field.from_integer(0))
        .expect("y^2 = x^3 + x is nonsingular");
    println!("\n2. The starting curve");
    println!(
        "   y^2 = x^3 + x, with j = {} and {} points",
        show(p, &start.j_invariant()),
        start.order()
    );
    println!(
        "   {p} = 3 (mod 4) makes it supersingular: the trace is {}, which p divides,",
        start.trace_of_frobenius()
    );
    println!("   and an isogeny cannot change that, so the walk stays in the graph.");

    // Step 3. Walk outwards.
    println!("\n3. The walk");
    let (vertices, edges) = explore(p, &start);

    // Step 4. What came out. Multiple edges between one pair are kept, so every line
    // lists three targets.
    println!("\n4. The adjacency");
    for (from, vertex) in vertices.iter().enumerate() {
        let targets: Vec<String> = edges
            .iter()
            .filter(|(tail, _)| *tail == from)
            .map(|(_, head)| show(p, &vertices[*head].j_invariant))
            .collect();
        let label = show(p, &vertex.j_invariant);
        println!("   j = {label:<14} -> {}", targets.join(", "));
    }

    // Step 5. The theory. Two of the five were checked during the walk, where what
    // they are about was in hand.
    println!("\n5. The checks");
    println!("   all three 2-torsion points of every curve reached lie in F_{p}^2");
    println!("   every codomain is supersingular, so the walk never left the graph");
    check_regularity(p, &vertices, &edges);
    check_vertex_count(p, &vertices);
    check_symmetry(p, &field, &vertices, &edges);
}

/// Breadth-first search from `start`, returning the vertices and the edges as pairs of
/// indices into them. A codomain whose j-invariant is new becomes the curve kept for
/// that class.
fn explore(p: u32, start: &Arc<Curve<Field>>) -> (Vec<Vertex>, Vec<(usize, usize)>) {
    let mut vertices = vec![Vertex {
        j_invariant: start.j_invariant(),
        curve: Arc::clone(start),
    }];
    let mut edges = Vec::new();

    let mut frontier = 0;
    while frontier < vertices.len() {
        let curve = Arc::clone(&vertices[frontier].curve);
        let (a, b) = curve.coefficients();
        println!(
            "   j = {}, reached as y^2 = x^3 + ({})x + ({})",
            show(p, &vertices[frontier].j_invariant),
            show(p, a),
            show(p, b)
        );

        // Check 1. A supersingular curve over F_{p^2} has all of its 2-torsion there,
        // so the cubic splits into three distinct roots. A twist has the same
        // property, so which model Vélu happened to produce does not matter.
        let kernels = two_torsion(&curve);
        assert_eq!(
            kernels.len(),
            3,
            "the cubic has {} roots, not the three the full 2-torsion needs",
            kernels.len()
        );

        for x in kernels {
            // The kernel of a 2-isogeny is {O, T} for a point T of order two, which is
            // where y = 0 and so where the cubic vanishes.
            let t = curve
                .point(x.clone(), curve.field().zero())
                .expect("the cubic vanishes at x, so (x, 0) is on the curve");
            let codomain = Arc::clone(curve.isogeny(&t, 2).codomain());

            // Check 2. Isogenous curves share a trace, so this must hold of every
            // curve the walk reaches. Counting points says so without assuming it.
            assert!(
                codomain.is_supersingular(),
                "the codomain of a 2-isogeny left the supersingular graph"
            );

            let j_invariant = codomain.j_invariant();
            let head = vertices
                .iter()
                .position(|vertex| vertex.j_invariant == j_invariant)
                .unwrap_or_else(|| {
                    vertices.push(Vertex {
                        j_invariant: j_invariant.clone(),
                        curve: codomain,
                    });
                    vertices.len() - 1
                });
            println!(
                "     kernel at x = {} leads to j = {}",
                show(p, &x),
                show(p, &j_invariant)
            );
            edges.push((frontier, head));
        }
        frontier += 1;
    }
    (vertices, edges)
}

/// An element of `F_{p^2}`, which is a polynomial in `x` of degree at most one.
///
/// The library reduces representatives without canonicalising them, so one class can
/// arrive as `-4` and again as `19`. Printing those as two vertices would misreport
/// the graph, though the classes themselves compare equal either way.
fn show(p: u32, element: &Element) -> String {
    show_polynomial(p, element.representative())
}

/// A polynomial over `F_p`, highest degree first, with every coefficient taken into
/// `0..p`.
fn show_polynomial(p: u32, polynomial: &Polynomial<PrimeField>) -> String {
    let modulus = BigInt::from(p);
    let terms: Vec<String> = (0..=polynomial.degree().unwrap_or(0))
        .rev()
        .filter_map(|i| {
            let residue: BigInt = polynomial.coefficient(i).representative().clone().into();
            let c = ((residue % &modulus) + &modulus) % &modulus;
            match (c.to_string().as_str(), i) {
                ("0", _) => None,
                (c, 0) => Some(c.to_string()),
                ("1", 1) => Some("x".to_string()),
                ("1", i) => Some(format!("x^{i}")),
                (c, 1) => Some(format!("{c}x")),
                (c, i) => Some(format!("{c}x^{i}")),
            }
        })
        .collect();
    if terms.is_empty() {
        "0".to_string()
    } else {
        terms.join(" + ")
    }
}

/// The roots of `x^3 + ax + b`, by trying every element of the field: the
/// x-coordinates of the points of order two.
fn two_torsion(curve: &Arc<Curve<Field>>) -> Vec<Element> {
    let field = curve.field();
    let order = field.order();
    let mut roots = Vec::new();
    let mut index = BigInt::from(0u32);
    while index < order {
        let x = field.element_at(&index);
        if field.is_zero(&curve.evaluate(&x)) {
            roots.push(x);
        }
        index += 1u32;
    }
    roots
}

/// Check 3. Three points of order two give three kernels, so three edges leave every
/// vertex, counted with multiplicity.
fn check_regularity(p: u32, vertices: &[Vertex], edges: &[(usize, usize)]) {
    for (from, vertex) in vertices.iter().enumerate() {
        let out_degree = edges.iter().filter(|(tail, _)| *tail == from).count();
        assert_eq!(
            out_degree,
            3,
            "j = {} has out-degree {out_degree}",
            show(p, &vertex.j_invariant)
        );
    }
    println!("   every vertex has out-degree three, multiple edges counted separately");
}

/// Check 4. Deuring's count. The graph is connected, so reaching exactly that many
/// vertices also says the walk saw all of it.
fn check_vertex_count(p: u32, vertices: &[Vertex]) {
    let correction = match p % 12 {
        1 => 0,
        5 | 7 => 1,
        _ => 2,
    };
    let expected = (p / 12 + correction) as usize;
    assert_eq!(
        vertices.len(),
        expected,
        "the walk reached {} vertices, not the {expected} that exist",
        vertices.len()
    );
    println!(
        "   {expected} vertices, which is floor({p}/12) + {correction}, so the walk saw\n\
         \x20  every supersingular class: the graph is connected, so that is all of it"
    );
}

/// Check 5. A 2-isogeny has a dual 2-isogeny, so edges come in pairs. The exception is
/// the two vertices with extra automorphisms, `j = 0` and `j = 1728`, where the
/// automorphisms identify kernels and the counts need not match.
fn check_symmetry(p: u32, field: &Field, vertices: &[Vertex], edges: &[(usize, usize)]) {
    let exceptional = [field.from_integer(0), field.from_integer(1728)];
    let is_exceptional = |index: usize| exceptional.contains(&vertices[index].j_invariant);
    let count = |from: usize, to: usize| edges.iter().filter(|edge| **edge == (from, to)).count();

    let mut pairs = 0;
    for from in 0..vertices.len() {
        for to in 0..vertices.len() {
            if is_exceptional(from) || is_exceptional(to) {
                continue;
            }
            pairs += 1;
            assert_eq!(
                count(from, to),
                count(to, from),
                "j = {} and j = {} disagree on how many edges join them",
                show(p, &vertices[from].j_invariant),
                show(p, &vertices[to].j_invariant)
            );
        }
    }
    let total = vertices.len() * vertices.len();
    println!("   edges pair up in {pairs} of the {total} ordered pairs of vertices, as duals");
    println!("   require; the rest touch j = 0 or j = 1728, where automorphisms intervene");
}
