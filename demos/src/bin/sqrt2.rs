//! Demo: computing a square root as a constructive real.
//!
//! In the spirit of the sample program shipped with Hans-J. Boehm's Java
//! constructive-reals library. Run with:
//!
//! ```text
//! cargo run --example sqrt2
//! ```
//!
//! A [`ConstructiveReal`] holds no digits of its own — it is a recipe that can
//! be evaluated to any requested precision on demand. Below, the *same*
//! `root2` value is asked for 10, then 100, then 500 decimal places.

use ruffini::constructive_reals::ConstructiveReal;

fn main() {
    let root2 = ConstructiveReal::from_int(2).sqrt();

    println!("sqrt(2), evaluated to increasing precision from one value:");
    for digits in [10u32, 50, 100, 500] {
        println!("  {:3} digits: {}", digits, root2.to_decimal(digits));
    }

    // The constructive real is exact: it carries the true value of sqrt(2),
    // so squaring it and printing yields 2 with no accumulated error.
    let squared = root2.clone() * root2.clone();
    println!();
    println!("sqrt(2) * sqrt(2), to 40 decimal places (exactly 2):");
    println!("  {}", squared.to_decimal(40));

    // Comparison is precision-bounded: it answers "undetermined" rather than
    // looping forever when two values might be equal.
    println!();
    match root2.compare(&ConstructiveReal::from_int(1), -1000) {
        Some(ord) => println!("sqrt(2) vs 1: {:?}", ord),
        None => println!("sqrt(2) vs 1: undetermined within the precision bound"),
    }
}
