# ruffini

Algebraic structures over arbitrary-precision integers, written so that an algorithm
reads the way the mathematics does.

## What's in it

Each structure is generic over the others, so they stack: `F_p`, `F_p[x]`, `F_p[x][y]`,
matrices over `Q(x)`, multivariate polynomials over a quotient ring.

### Structures

- **`Integers`** — `Z` over `BigInt`, a Euclidean domain.
- **`QuotientRing`** — `ring.quotient(m)` is `R/(m)`, and `Integers::modulo(7)` is `F_7`.
- **`FractionField`** — `ring.fractions()` is the field of fractions, so
  `Integers::rationals()` is `Q` and `f.polynomials().fractions()` is `F(x)`.
- **`finite_field(p, k)`** — `F_{p^k}`, as `F_p[x]` modulo an irreducible polynomial
  found for you, with a table fixing the common `p` and `k`.
- **`PolynomialRing`** — `ring.polynomials()` is `R[x]`, stored densely, itself a
  Euclidean domain when `R` is a field.
- **`MultivariatePolynomialRing`** — `ring.multi_polynomials(3)` is
  `R[x_0, x_1, x_2]`, stored sparsely, with the variable count a run-time value.
- **`MatrixRing`** — `ring.matrices(n)` is the `n x n` matrices over `R`, itself a ring;
  `Matrix` on its own is any shape.
- **`Curve`** — elliptic curves `y² = x³ + ax + b` over any field, whose points form an
  additive group; over a finite one, `is_isomorphic` tells a curve from its twists.
- **`MontgomeryCurve`** — `y² = x³ + Ax² + x`, carrying a point as its `x` alone, in
  `(X : Z)`, where doubling, differential addition and the ladder never mention `y`;
  `to_weierstrass` moves to the other model.
- **`ConstructiveReals`** — computable reals, held as expression trees and evaluated to
  whatever precision is asked of them.

### Algorithms

- **Gröbner bases** — `gröbner::gröbner_basis`, Buchberger with both of his criteria,
  under any `MonomialOrder` (`Lex`, `GradedLex`), returning the reduced basis.
- **Division** — Euclidean for polynomials over a field, `div_rem_monic` by a monic
  divisor over any ring, and `MultivariatePolynomial::divide` by several divisors at once.
- **Extended gcd** — `euclidean::extended_gcd` over any Euclidean domain, canonicalised:
  positive over `Z`, monic over `F[x]`.
- **Interpolation** — `interpolation::interpolate`, or `Interpolation::new` to reuse the
  Lagrange basis across several sets of values.
- **Determinant and inverse** — cofactor expansion over any ring, Gauss-Jordan over a
  field.
- **Fast Fourier transform** — `fft::fft` and `inverse_fft`, radix-2 over any ring that
  has a root of unity.
- **Exponentiation** — `x.pow(n)` by square-and-multiply, the exponent any integer type
  up to `BigInt`.
- **Multi-point evaluation** — `p.evaluations(start, step)` walks an arithmetic
  progression by forward differences, at one addition per coefficient per point and no
  multiplication.
- **Number theory** — `number_theory::factorise`, `totient` and `multiplicative_order`,
  by trial division, sized for the small moduli that turn up as parameters.
- **Roots** — `field.root(&x, k)` for any `k`, by Adleman, Manders and Miller one prime
  factor at a time, over any `FiniteField`, with `roots` for all `gcd(k, q-1)` of them;
  `sqrt` is the case `k = 2` and Tonelli and Shanks, `root_of_unity` gives a primitive
  one, and `legendre_symbol` and `is_kth_power` answer the existence questions alone.
- **Elliptic curve arithmetic** — the chord-and-tangent group law and scalar
  multiplication by double-and-add.
- **Point counting** — `curve.order()` over a finite field, by summing the quadratic
  character of `x³ + ax + b`, with `trace_of_frobenius` for `q + 1 - #E` and
  `is_supersingular`.
- **Division polynomials** — `curve.division_polynomials(n)`, held as `gₙ` with
  `ψₙ = gₙ` for odd `n` and `y·gₙ` for even, so they stay in `F[x]`;
  `multiplication_map(n)` gives multiplication by `n` on `x` as `φₙ/ψₙ²`.
- **Isogenies** — `curve.isogeny(&generator, order)` by Vélu's formulas: the separable
  isogeny with the given kernel, with its codomain, `evaluate` on points, and `x_map`
  for `r₁` in `F(x)`.
- **Isomorphisms** — `curve.isomorphism(&other)` finds the `u` of
  `(x, y) ↦ (u²x, u³y)` over the field itself, `isomorphisms` gives all `gcd(k, q-1)`
  of them, and `is_isomorphic` answers the question alone.
- **Dual isogenies** — `isogeny.dual()` over a finite field, by Vélu on the image of
  the `n`-torsion and then the isomorphism that makes the composite `[n]`;
  `isogeny.then(iso)` composes an isomorphism onto an isogeny.

### Notation

- Operators on elements, in every combination of owned and borrowed operands: `a + b`,
  `&a * &b`, `a - &b`, and `+=` / `*=`.
- Plain integers on either side of an operator: `2 * x`, `x - 1`.
- Polynomials built from their indeterminate: `let x = ring.indeterminate();` and then
  `x.pow(7) - 1`.
- Elements are `Send` and `Sync` whenever their coefficients are.

## Finite fields

```rust
use ruffini::integers::Integers;
use ruffini::structures::{Field, Monoid};

let f7 = Integers::modulo(7);

assert_eq!(f7.inverse(3).unwrap() * 3, f7.identity()); // 3 · 3⁻¹ = 1
```

## Finite fields of prime power order

`F_{p^k}` is `F_p[x]` modulo an irreducible polynomial of degree `k`, which the library
finds, tests with Rabin's criterion, or looks up for small `p` and `k`.

```rust
use ruffini::finite_fields::finite_field;
use ruffini::structures::{Domain, Field, Monoid};

let f9 = finite_field(3, 2); // F_9 = F_3[x]/(x^2 + 1)
let ring = f9.base();        // F_3[x], to build elements with

let i = f9.element(ring.indeterminate()); // x, a square root of -1
assert_eq!(i.pow(2), -f9.identity());
assert_eq!(i.pow(8), f9.identity());      // every nonzero a has a^(9-1) = 1

assert_eq!(&i * &f9.invert(&i).unwrap(), f9.identity());
```

## Polynomials

Written from the indeterminate, with integers on either side of an operator, rather
than from a vector of coefficients.

```rust
use ruffini::integers::Integers;
use ruffini::polynomials::RingExt;

let zx = Integers::default().polynomials(); // Z[x]
let x = zx.indeterminate();

let modulus = x.pow(7) - 1;        // x⁷ - 1
let p = 2 * x.pow(3) + 3 * x - 1;  // 2x³ + 3x - 1

assert_eq!(p.evaluate(&2.into()), 21.into());
```

## Gröbner bases

Buchberger's algorithm, under any monomial order, returning the reduced basis. Dividing
by it decides membership of the ideal.

```rust
use ruffini::gröbner::gröbner_basis;
use ruffini::integers::Integers;
use ruffini::multivariate::Lex;
use ruffini::polynomials::RingExt;
use ruffini::structures::CommutativeMonoid;

let r = Integers::modulo(7).multi_polynomials(2);
let (x, y) = (r.variable(0), r.variable(1));

// The circle meeting the line x = y, so 2y² = 1 and y² = 4.
let basis = gröbner_basis(&r, &[x.pow(2) + y.pow(2) - 1, x.clone() - y.clone()], &Lex);
assert_eq!(basis, vec![x.clone() - y.clone(), y.pow(2) + 3]);

assert!(r.is_zero(&(x.pow(2) + y.pow(2) - 1).divide(&basis, &Lex).1));
```

## Constructive reals

A value is a recipe rather than digits, evaluated to whatever precision is asked of it.

```rust
use ruffini::constructive_reals::ConstructiveReal;

let root2 = ConstructiveReal::from_int(2).sqrt();

assert_eq!(root2.to_decimal(10), "1.4142135624");
assert_eq!((root2.clone() * root2).to_decimal(10), "2.0000000000");
```

## Demos

In their own crate, so that what they need - a random number generator, for one - stays
out of the library's dependencies.

```text
cargo run --release -p ruffini-demos --bin sqrt2          # constructive reals
cargo run --release -p ruffini-demos --bin hadamard -- 23 # Hadamard matrix of order 92
cargo run --release -p ruffini-demos --bin aks            # AKS primality test
cargo run --release -p ruffini-demos --bin shamir         # Shamir secret sharing
cargo run --release -p ruffini-demos --bin isogeny_graph # supersingular 2-isogeny graph
cargo run --release -p ruffini-demos --bin cgl_hash      # the CGL hash
```

## Build & test

```text
cargo test
```

## License

MIT. Named for [Paolo Ruffini](https://en.wikipedia.org/wiki/Paolo_Ruffini).
