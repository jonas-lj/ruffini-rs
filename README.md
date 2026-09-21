# ruffini

A small Rust library exploring algebraic structures over arbitrary-precision
integers.

## What's in it

A trait hierarchy mirroring the standard algebraic tower:

```text
Domain → Semigroup → Monoid → SemiRing → Ring → EuclideanDomain → Field
              ↘                   ↗
       CommutativeMonoid → AdditiveGroup
```

…together with three concrete instances:

- **`Integers`** (`Z`) over `num_bigint::BigInt` — a Euclidean domain.
- **`QuotientRing<R>`** — the quotient `R / (m)`. With a prime modulus this is
  a finite prime field `F_p` (the `Field` impl is in place; primality is a
  caller-supplied invariant).
- **`PolynomialRing<R>`** — `R[x]`. When `R: Field` this is itself a Euclidean
  domain, so you can take quotients of polynomial rings by irreducible
  polynomials and chain the construction up to `F_{p^k}`.
- **`ConstructiveReals`** — computable reals, held lazily as expression trees and
  evaluated to any requested precision. Equality is undecidable, so they form a
  `Ring` but not a `Field`; see `cargo run --example sqrt2`.

`Domain::E` does not require `Eq` — only `EuclideanDomain` and `Field` do, since
those are where an algorithm tests for zero.

`Field::invert` and the generic `extended_gcd` both rely on
`EuclideanDomain::unit_part` / `unit_inverse` to canonicalise gcds — non-negative
for `Z`, monic for `R[x]`.

## Example

```rust
use ruffini::integers::Integers;
use ruffini::structures::{Field, Monoid};

let f7 = Integers::modulo(7);

assert_eq!(f7.inverse(3).unwrap() * 3, f7.identity()); // 3 · 3⁻¹ = 1 in F_7
```

Integers are embedded into the ring on the fly, via `Ring::from_integer` (the
canonical map `n ↦ n · 1`, by double-and-add). To name an element explicitly,
use `Domain::element`:

```rust
use ruffini::integers::Integers;
use ruffini::polynomials::RingExt;
use ruffini::structures::Domain;

let f7 = Integers::modulo(7);
assert_eq!(&f7.element(3) + &f7.element(6), f7.element(2)); // 9 ≡ 2 (mod 7)

let f7x = f7.polynomials(); // F_7[x], and f7x.polynomials() is F_7[x][y]
assert_eq!(f7x.element(vec![f7.element(1), f7.element(2)]).degree(), Some(1));
```

## Build & test

```text
cargo test
```

Named for [Paolo Ruffini](https://en.wikipedia.org/wiki/Paolo_Ruffini).
