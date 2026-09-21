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
