//! Operator boilerplate shared between element types.

/// Arithmetic between an element and a plain integer, for one operator and one integer
/// type. The integer is embedded with [`Ring::from_integer`](crate::structures::Ring)
/// first, so `x * 3` means `x * (3 · 1)` in `x`'s own ring.
///
/// The element must be generic over exactly one parameter `R` and hold a `ring` field.
macro_rules! int_operand_op {
    ($elem:ty, { $($bounds:tt)* }, $int:ty, $op:ident, $method:ident) => {
        impl<R> $op<$int> for $elem
        where
            $($bounds)*
        {
            type Output = $elem;
            fn $method(self, rhs: $int) -> Self::Output {
                let rhs = self.ring.from_integer(rhs);
                self.$method(rhs)
            }
        }

        impl<R> $op<$int> for &$elem
        where
            $($bounds)*
            for<'c> &'c $elem: $op<&'c $elem, Output = $elem>,
        {
            type Output = $elem;
            fn $method(self, rhs: $int) -> Self::Output {
                let rhs = self.ring.from_integer(rhs);
                self.$method(&rhs)
            }
        }
    };
}

/// [`int_operand_op`] across `Add`, `Sub` and `Mul` for each integer type given.
macro_rules! int_operand_ops {
    ($elem:ty, $bounds:tt, $($int:ty),+ $(,)?) => {$(
        int_operand_op!($elem, $bounds, $int, Add, add);
        int_operand_op!($elem, $bounds, $int, Sub, sub);
        int_operand_op!($elem, $bounds, $int, Mul, mul);
    )+};
}

/// The two mixed owned/borrowed combinations of one operator, forwarded to the fully
/// borrowed impl, which the element type supplies itself.
macro_rules! forward_ref_binop {
    ($elem:ty, { $($bounds:tt)* }, $op:ident, $method:ident) => {
        impl<R> $op<&$elem> for $elem
        where
            $($bounds)*
            for<'c> &'c $elem: $op<&'c $elem, Output = $elem>,
        {
            type Output = $elem;
            fn $method(self, rhs: &$elem) -> Self::Output {
                (&self).$method(rhs)
            }
        }

        impl<R> $op<$elem> for &$elem
        where
            $($bounds)*
            for<'c> &'c $elem: $op<&'c $elem, Output = $elem>,
        {
            type Output = $elem;
            fn $method(self, rhs: $elem) -> Self::Output {
                self.$method(&rhs)
            }
        }
    };
}

/// [`forward_ref_binop`] for each operator given.
macro_rules! forward_ref_binops {
    ($elem:ty, $bounds:tt, $($op:ident, $method:ident);* $(;)?) => {$(
        forward_ref_binop!($elem, $bounds, $op, $method);
    )*};
}

/// The mirror of [`int_operand_op`]: the integer on the left. One impl per integer
/// type, since `impl<T: Into<BigInt>> Mul<Elem> for T` is not allowed.
///
/// Owned operands only. A `&Elem` version compiles, but it reaches the higher-ranked
/// `&'c Elem: Mul<&'c Elem>` bound, and inference then drags that recursion into
/// unrelated integer arithmetic until it overflows.
macro_rules! scalar_operand_op {
    ($elem:ty, { $($bounds:tt)* }, $int:ty, $op:ident, $method:ident) => {
        impl<R> $op<$elem> for $int
        where
            $($bounds)*
        {
            type Output = $elem;
            fn $method(self, rhs: $elem) -> Self::Output {
                let lhs = rhs.ring.from_integer(self);
                lhs.$method(rhs)
            }
        }
    };
}

/// [`scalar_operand_op`] across `Add`, `Sub` and `Mul` for each integer type given.
macro_rules! scalar_operand_ops {
    ($elem:ty, $bounds:tt, $($int:ty),+ $(,)?) => {$(
        scalar_operand_op!($elem, $bounds, $int, Add, add);
        scalar_operand_op!($elem, $bounds, $int, Sub, sub);
        scalar_operand_op!($elem, $bounds, $int, Mul, mul);
    )+};
}

/// `pow` as a method on an element that carries its own ring, so exponentiation reads
/// as the notation does: `x.pow(n)` rather than `pow(&ring, &x, n)`.
///
/// The element must hold a `ring` field whose handle is a [`Monoid`](crate::structures::Monoid).
macro_rules! element_pow {
    ($elem:ty, { $($bounds:tt)* }) => {
        impl<R> $elem
        where
            $($bounds)*
        {
            /// `self^exp`, in `O(log exp)` multiplications. `exp == 0` gives one.
            ///
            /// # Panics
            /// If `exp` is negative.
            pub fn pow<T: Into<::num_bigint::BigInt>>(&self, exp: T) -> Self {
                $crate::pow::pow(&self.ring, self, exp)
            }
        }
    };
}
