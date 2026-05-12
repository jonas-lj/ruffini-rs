# ruffini

A small Rust library exploring algebraic structures over arbitrary-precision
integers.

## What's in it

A trait hierarchy mirroring the standard algebraic tower:

```
Set → Semigroup → Monoid → SemiRing → Ring → EuclideanDomain → Field
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

The `Field::inverse` and generic `extended_gcd` both rely on
`EuclideanDomain::unit_part` / `unit_inverse` to canonicalise gcds — non-negative
for `Z`, monic for `R[x]`.

## Example

```rust
use std::rc::Rc;
use num_bigint::BigInt;
use ruffini::integers::{Integer, Integers};
use ruffini::structures::{Field, Monoid, QuotientRing, QuotientRingElement};

let f7 = QuotientRing::new(Integers::default(), Integer::from(BigInt::from(7)));
let three = QuotientRingElement::new(&f7, Integer::from(BigInt::from(3)));

assert_eq!(f7.inverse(&three).unwrap() * three, f7.identity()); // 3 · 3⁻¹ = 1 in F_7
```

## Build & test

```
cargo test
```

Named for [Paolo Ruffini](https://en.wikipedia.org/wiki/Paolo_Ruffini).
