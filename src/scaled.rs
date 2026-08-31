//! Narrow scale helpers for finite solver arithmetic.
//!
//! [`product_ratio`] combines the few scale factors used to normalize Jacobians
//! and physical updates without depending on multiplication order. [`ScaledSum`]
//! is reserved for SVD solution reconstruction, where independently verified
//! finite answers can require cancellation between singular contributions
//! outside the `f64` exponent range. Ordinary dot products deliberately use
//! [`CompensatedSum`] and report ordinary floating-point range failure.

/// Neumaier-compensated sum using ordinary `f64` range and precision.
///
/// Compensation recovers low-order rounding lost during finite additions; it
/// does not extend the exponent range. A non-finite term or partial sum remains
/// non-finite for the caller to reject at its numerical boundary.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CompensatedSum {
    /// Ordinary running total.
    sum: f64,
    /// Low-order rounding residual accumulated separately from `sum`.
    correction: f64,
}

impl CompensatedSum {
    /// Add one value while retaining the rounding residue of the larger addend.
    pub(crate) fn add(&mut self, value: f64) {
        let next = self.sum + value;
        self.correction += if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        self.sum = next;
    }

    /// Return the compensated result in ordinary `f64` range.
    pub(crate) fn total(self) -> f64 {
        self.sum + self.correction
    }
}

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

/// Finite scalar retained as a signed normalized significand and an independent
/// binary exponent.
///
/// Keeping this representation private lets matrix and solver reductions carry
/// an intermediate value beyond the ordinary `f64` exponent range without
/// claiming an arbitrary-precision public number type. Only the completed
/// result is rounded back to `f64`.
#[derive(Debug, Clone, Copy)]
struct ScaledValue {
    /// Signed significand whose magnitude is zero or lies in `[1, 2)`.
    significand: f64,
    /// Unbiased power of two applied to `significand`.
    exponent: i64,
}

impl ScaledValue {
    /// Exact additive identity in the scaled representation.
    const ZERO: Self = Self {
        significand: 0.0,
        exponent: 0,
    };

    /// Normalize one signed significand/exponent pair into the representation's
    /// stable magnitude interval.
    fn normalized(mut significand: f64, mut exponent: i64) -> Self {
        debug_assert!(significand.is_finite());
        if significand == 0.0 {
            // Canonicalizing signed zero makes accumulator initialization and
            // exact cancellation independent of operand signs.
            return Self::ZERO;
        }

        // Factor counts are deliberately small, but normalization here also
        // handles a compensated sum whose magnitude grows beyond two or falls
        // below one after cancellation.
        while significand.abs() >= 2.0 {
            significand *= 0.5;
            exponent += 1;
        }
        while significand.abs() < 1.0 {
            significand *= 2.0;
            exponent -= 1;
        }
        Self {
            significand,
            exponent,
        }
    }

    /// Form one product divided by another product without materializing a
    /// range-sensitive intermediate `f64`.
    fn product_ratio<const N: usize, const D: usize>(
        numerators: [f64; N],
        denominators: [f64; D],
    ) -> Self {
        let mut significand = 1.0;
        let mut exponent = 0_i64;
        let mut negative = false;

        for value in numerators {
            debug_assert!(value.is_finite());
            if value == 0.0 {
                // All call sites establish finite non-zero denominators, so a
                // zero numerator makes the complete term exactly zero.
                return Self::ZERO;
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

        if negative {
            significand = -significand;
        }
        Self::normalized(significand, exponent)
    }

    /// Round this completed scaled value to the nearest representable `f64`.
    fn to_f64(self) -> f64 {
        if self.significand == 0.0 {
            return 0.0;
        }
        let magnitude = compose_positive(self.significand.abs(), self.exponent);
        if self.significand.is_sign_negative() {
            -magnitude
        } else {
            magnitude
        }
    }
}

/// Compensated sum whose terms are aligned by binary exponent before addition.
///
/// Each term can itself be a product or quotient represented by [`ScaledValue`].
/// The accumulator therefore avoids both forms of order-dependent range loss:
/// overflow while forming one term and overflow in a partial sum that later
/// cancellation would have returned to the finite range.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScaledSum {
    /// Common exponent to which `sum` and `correction` are aligned.
    exponent: i64,
    /// Ordinary running sum of bounded, exponent-aligned significands.
    sum: f64,
    /// Neumaier compensation retaining low-order rounding residuals.
    correction: f64,
    /// Whether a non-zero term established `exponent`.
    initialized: bool,
}

impl Default for ScaledSum {
    /// Construct an empty accumulator representing exact zero.
    fn default() -> Self {
        Self {
            exponent: 0,
            sum: 0.0,
            correction: 0.0,
            initialized: false,
        }
    }
}

impl ScaledSum {
    /// Add one scaled value after aligning it with the accumulator's reference
    /// exponent.
    fn add_scaled(&mut self, value: ScaledValue) {
        if value.significand == 0.0 {
            // Exact zero has no exponent and cannot affect compensation.
            return;
        }

        let term = if !self.initialized {
            self.exponent = value.exponent;
            self.initialized = true;
            value.significand
        } else if value.exponent > self.exponent {
            // Rescale prior state downward before adopting the larger exponent.
            // The exponent difference is non-positive and therefore cannot
            // overflow while it is materialized as a power of two.
            let prior_scale = compose_positive(1.0, self.exponent - value.exponent);
            self.sum *= prior_scale;
            self.correction *= prior_scale;
            self.exponent = value.exponent;
            value.significand
        } else {
            // Align a smaller term to the existing scale. Values more than the
            // subnormal range below it round away exactly as ordinary `f64`
            // precision requires, without overflowing the dominant partial sum.
            let term_scale = compose_positive(1.0, value.exponent - self.exponent);
            value.significand * term_scale
        };

        // Neumaier compensation keeps the rounding residue from whichever of
        // the existing sum and new term has greater magnitude.
        let next = self.sum + term;
        self.correction += if self.sum.abs() >= term.abs() {
            (self.sum - next) + term
        } else {
            (term - next) + self.sum
        };
        self.sum = next;
    }

    /// Form and add one finite product ratio through the shared scaled-number
    /// representation.
    pub(crate) fn add_product_ratio<const N: usize, const D: usize>(
        &mut self,
        numerators: [f64; N],
        denominators: [f64; D],
    ) {
        self.add_scaled(ScaledValue::product_ratio(numerators, denominators));
    }

    /// Return the compensated result while preserving its binary exponent.
    fn scaled_total(self) -> ScaledValue {
        if !self.initialized {
            return ScaledValue::ZERO;
        }
        ScaledValue::normalized(self.sum + self.correction, self.exponent)
    }

    /// Round the completed compensated sum to `f64` exactly once.
    pub(crate) fn total(self) -> f64 {
        self.scaled_total().to_f64()
    }
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
    // Preserve the scalar convenience used by non-reduction call sites while
    // routing its implementation through the same representation as sums.
    ScaledValue::product_ratio(numerators, denominators).to_f64()
}

#[cfg(test)]
mod tests {
    use super::{CompensatedSum, ScaledSum, product_ratio};

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

    /// Distinguish rounding compensation from exponent-range extension.
    #[test]
    fn compensated_sum_recovers_roundoff_but_preserves_f64_range_failure() {
        let mut rounded = CompensatedSum::default();
        rounded.add(1e16);
        rounded.add(1.0);
        rounded.add(-1e16);
        assert_eq!(rounded.total(), 1.0);

        // The first addition overflows in ordinary evaluation order. A later
        // cancellation does not retroactively make that operation valid.
        let mut overflowed = CompensatedSum::default();
        overflowed.add(f64::MAX);
        overflowed.add(f64::MAX);
        overflowed.add(-f64::MAX);
        assert!(!overflowed.total().is_finite());
    }

    /// Ensure partial sums and even individual terms can temporarily exceed the
    /// `f64` exponent range when later cancellation leaves a finite answer.
    #[test]
    fn scaled_sum_delays_range_checks_until_the_completed_reduction() {
        let mut partial_overflow = ScaledSum::default();
        partial_overflow.add_product_ratio([f64::MAX], []);
        partial_overflow.add_product_ratio([f64::MAX], []);
        partial_overflow.add_product_ratio([-f64::MAX], []);
        assert_eq!(partial_overflow.total(), f64::MAX);

        // Each of the first two products is individually outside `f64`, but
        // their exact cancellation must not erase the final unit term.
        let mut term_overflow = ScaledSum::default();
        term_overflow.add_product_ratio([f64::MAX, 2.0], []);
        term_overflow.add_product_ratio([-f64::MAX, 2.0], []);
        term_overflow.add_product_ratio([1.0], []);
        assert_eq!(term_overflow.total(), 1.0);
    }
}
