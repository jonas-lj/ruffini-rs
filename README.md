# ruffini

Algebraic structures over arbitrary-precision integers, written so that an algorithm
reads the way the mathematics does.

Rings, fields, polynomials in one or many variables, matrices, elliptic curves and
constructive reals, each generic over the others.

## Finite fields

```rust
use ruffini::integers::Integers;
use ruffini::structures::{Field, Monoid};

let f7 = Integers::modulo(7);

assert_eq!(f7.inverse(3).unwrap() * 3, f7.identity()); // 3 · 3⁻¹ = 1
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

```text
cargo run --release --example sqrt2             # constructive reals
cargo run --release --example hadamard -- 23    # Hadamard matrix of order 92
cargo run --release --example aks               # AKS primality test
```

## Build & test

```text
cargo test
```

## License

MIT. Named for [Paolo Ruffini](https://en.wikipedia.org/wiki/Paolo_Ruffini).
