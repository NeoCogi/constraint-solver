//! Scale-safe products and quotients for finite solver arithmetic.
//!
//! Numerical normalization repeatedly combines values whose individual units
//! can span the complete `f64` exponent range. Evaluating those expressions in
//! an arbitrary source-code order can overflow or underflow even when the final
//! mathematical result is representable. This module supplies one private
//! policy that combines significands and binary exponents separately.

/// Fraction bits stored in one IEEE-754 binary64 value.
const FRACTION_BITS: u32 = 52;
/// Mask selecting the stored binary64 fraction field.
const FRACTION_MASK: u64 = (1_u64 << FRACTION_BITS) - 1;
/// Stored exponent that represents an unbiased exponent of zero.
const EXPONENT_BIAS: i64 = 1023;
/// Exact factor that promotes every non-zero subnormal into the normal range.
const SUBNORMAL_SCALE: f64 = (1_u64 << FRACTION_BITS) as f64;

/// Split one positive finite value into a significand in `[1, 2)` and an
/// unbiased binary exponent.
fn decompose_positive(value: f64) -> (f64, i64) {
    debug_assert!(value.is_finite() && value > 0.0);

    let bits = value.to_bits();
    let stored_exponent = ((bits >> FRACTION_BITS) & 0x7ff) as i64;
    if stored_exponent != 0 {
        // Re-bias the stored fraction to exponent zero. The resulting value is
        // exactly the significand because only exponent bits are changed.
        let significand_bits = ((EXPONENT_BIAS as u64) << FRACTION_BITS) | (bits & FRACTION_MASK);
        return (
            f64::from_bits(significand_bits),
            stored_exponent - EXPONENT_BIAS,
        );
    }

    // A subnormal has no implicit leading one. Multiplication by 2^52 is exact
    // and moves even the smallest subnormal into the normal range, where the
    // ordinary decomposition above applies. Subtract the promotion exponent
    // from the returned exponent to preserve the original value.
    let promoted = value * SUBNORMAL_SCALE;
    let promoted_bits = promoted.to_bits();
    let promoted_exponent = ((promoted_bits >> FRACTION_BITS) & 0x7ff) as i64;
    let significand_bits =
        ((EXPONENT_BIAS as u64) << FRACTION_BITS) | (promoted_bits & FRACTION_MASK);
    (
        f64::from_bits(significand_bits),
        promoted_exponent - EXPONENT_BIAS - i64::from(FRACTION_BITS),
    )
}

/// Reassemble a non-negative normalized significand and binary exponent.
fn compose_positive(significand: f64, exponent: i64) -> f64 {
    debug_assert!((1.0..2.0).contains(&significand));

    if exponent > 1023 {
        // The correctly scaled magnitude lies beyond the largest normal
        // exponent, so the mathematical result is not representable.
        return f64::INFINITY;
    }
    if exponent == -1075 {
        // This interval straddles the halfway point below the smallest
        // subnormal. Exact halfway rounds to the even zero representation;
        // every larger significand rounds up to the smallest subnormal.
        return if significand > 1.0 {
            f64::from_bits(1)
        } else {
            0.0
        };
    }
    if exponent < -1075 {
        // Values below half of the subnormal range round to zero. Returning
        // zero also exposes genuine underflow to the solver's boundary checks.
        return 0.0;
    }

    let binary_scale = if exponent >= -1022 {
        // Construct an exact normal power of two directly from its biased
        // exponent so no library `pow` implementation participates.
        f64::from_bits(((exponent + EXPONENT_BIAS) as u64) << FRACTION_BITS)
    } else {
        // Subnormal powers from 2^-1074 through 2^-1023 are individual fraction
        // bits. Multiplying by the normalized significand performs the final
        // IEEE-754 rounding at the requested subnormal scale.
        let fraction_bit = (exponent + 1074) as u32;
        f64::from_bits(1_u64 << fraction_bit)
    };
    significand * binary_scale
}

/// Evaluate a product divided by another product without order-dependent
/// intermediate overflow or underflow.
///
/// Every input must be finite and every denominator must be non-zero. Those
/// invariants are established at the public solver and matrix boundaries. A
/// zero numerator returns zero immediately; otherwise, the function combines
/// bounded significands and integer exponents before performing one final
/// floating-point scaling operation.
pub(crate) fn product_ratio<const N: usize, const D: usize>(
    numerators: [f64; N],
    denominators: [f64; D],
) -> f64 {
    let mut significand = 1.0;
    let mut exponent = 0_i64;
    let mut negative = false;

    for value in numerators {
        debug_assert!(value.is_finite());
        if value == 0.0 {
            // All solver uses have finite non-zero denominators, so a zero
            // numerator makes the complete mathematical product exactly zero.
            return 0.0;
        }
        negative ^= value.is_sign_negative();
        let (factor_significand, factor_exponent) = decompose_positive(value.abs());
        significand *= factor_significand;
        exponent += factor_exponent;
    }

    for value in denominators {
        debug_assert!(value.is_finite() && value != 0.0);
        negative ^= value.is_sign_negative();
        let (factor_significand, factor_exponent) = decompose_positive(value.abs());
        significand /= factor_significand;
        exponent -= factor_exponent;
    }

    // At the small fixed arities used by the solver, these loops execute only a
    // handful of times. Keeping the significand normalized makes composition
    // independent of how many numerator or denominator factors a caller uses.
    while significand >= 2.0 {
        significand *= 0.5;
        exponent += 1;
    }
    while significand < 1.0 {
        significand *= 2.0;
        exponent -= 1;
    }

    let magnitude = compose_positive(significand, exponent);
    if negative { -magnitude } else { magnitude }
}

#[cfg(test)]
mod tests {
    use super::product_ratio;

    /// Compare non-zero values relatively so subnormal regressions cannot pass
    /// merely because an absolute tolerance is much larger than the answer.
    fn assert_relative_close(actual: f64, expected: f64, tolerance: f64) {
        let relative_error = (actual / expected - 1.0).abs();
        assert!(
            relative_error <= tolerance,
            "expected {expected:e}, got {actual:e}; relative error {relative_error:e}"
        );
    }

    /// Cover the multiplication and division associations used by Jacobian
    /// normalization, physical variable updates, and SVD scale restoration.
    #[test]
    fn product_ratio_avoids_representable_intermediate_range_failures() {
        // Multiplying the two numerator factors first overflows, while dividing
        // the two small factors first underflows.
        assert_relative_close(product_ratio([1e308, 1e-308], [1e308]), 1e-308, 1e-12);
        assert_relative_close(product_ratio([1e-320, 1e-12, 1e308], []), 1e-24, 1e-3);
        assert_relative_close(product_ratio([1e308, 1e-2, 10.0], []), 1e307, 1e-12);

        // Either sequential division order overflows or underflows for some
        // scale combinations; exponent accumulation keeps the final quotient.
        assert_relative_close(product_ratio([1.0], [1e-308, 1e308]), 1.0, 1e-12);
        assert_relative_close(product_ratio([1e308], [1e-308, 1e308]), 1e308, 1e-12);
    }
}
