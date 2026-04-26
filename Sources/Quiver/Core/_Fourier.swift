// Copyright 2026 Wayne W Bishop. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this
// file except in compliance with the License. You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under
// the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
// ANY KIND, either express or implied. See the License for the specific language governing
// permissions and limitations under the License.

import Foundation

// MARK: - Internal Complex Number Type

/// Internal representation of a complex number used by Fourier computation.
///
/// Kept internal following Quiver's `_Vector` / `_Regression` pattern — the public
/// API surfaces only plain `[Double]` results, so callers never interact with this type
/// directly. The underscore prefix signals that this is an implementation detail.
///
/// Complex arithmetic is the only genuinely new math in this file. Addition,
/// subtraction, and magnitude reuse the same formulas already present in Quiver's
/// vector operations. Only multiplication is specific to complex numbers:
///
///   (a + bi)(c + di) = (ac − bd) + (ad + bc)i
///
internal struct _Complex {

    var real: Double
    var imag: Double

    // MARK: - Arithmetic

    /// Complex addition — same formula as Quiver's existing `.add()`.
    func add(_ other: _Complex) -> _Complex {
        _Complex(real: real + other.real, imag: imag + other.imag)
    }

    /// Complex subtraction — same formula as Quiver's existing `.subtract()`.
    func subtract(_ other: _Complex) -> _Complex {
        _Complex(real: real - other.real, imag: imag - other.imag)
    }

    /// Complex multiplication.
    ///
    /// (a + bi)(c + di) = (ac − bd) + (ad + bc)i
    func multiply(_ other: _Complex) -> _Complex {
        _Complex(
            real: real * other.real - imag * other.imag,
            imag: real * other.imag + imag * other.real
        )
    }

    // MARK: - Derived Values

    /// Amplitude at this frequency bin.
    ///
    /// Identical formula to Quiver's `.magnitude` — √(real² + imag²) — applied
    /// to a single complex value rather than across an entire vector.
    var magnitude: Double {
        (real * real + imag * imag).squareRoot()
    }

    /// Phase angle in radians, in the range (−π, π].
    var phase: Double {
        Foundation.atan2(imag, real)
    }
}

// MARK: - Internal Fourier Engine

/// Internal namespace for Cooley-Tukey FFT computation.
///
/// Follows the `_Regression` / `_Metrics` / `_Sampling` pattern — all computation
/// lives here, the public `Array` extension is a thin wrapper that converts results
/// to plain `[Double]` before returning them to the caller.
///
/// **Algorithm overview**
///
/// The Cooley-Tukey radix-2 FFT is a divide-and-conquer algorithm in the same family
/// as Quicksort. Both algorithms:
///   - split the input in half at each level (Quicksort: left/right; FFT: even/odd)
///   - recurse on each half independently
///   - combine the two results in O(n) work per level
///   - achieve O(n log n) total through log n levels of recursion
///
/// The recurrence T(n) = 2T(n/2) + O(n) is identical to Quicksort's average case
/// and is solved by the Master Theorem to give O(n log n).
///
/// **Requirement: power-of-two input length**
///
/// The radix-2 algorithm requires the sample count to be a power of 2 (8, 16, 32, 64 …)
/// so that even/odd splitting is always exact. Use `nextPowerOfTwo(for:)` to pad a signal
/// before calling `transform(_:)` when the length is not already a power of 2.
internal enum _Fourier {

    // MARK: - Core Transform (Complex Input)

    /// Computes the forward Discrete Fourier Transform on complex-valued input
    /// using the Cooley-Tukey radix-2 algorithm.
    ///
    /// - Parameter signal: Complex-valued input samples. Length **must** be a power of 2.
    /// - Returns: Complex frequency-domain values of the same length.
    /// - Complexity: O(n log n) time, O(n log n) space for the recursion stack.
    static func transform(_ signal: [_Complex]) -> [_Complex] {

        let sampleCount = signal.count

        // Base case — a single sample has trivial frequency content
        guard sampleCount > 1 else {
            return signal
        }

        // ── DIVIDE ──────────────────────────────────────────────────────────
        // Split into even-indexed and odd-indexed samples.
        // This is the FFT's equivalent of Quicksort's partition step.
        let evenSamples = stride(from: 0, to: sampleCount, by: 2).map { signal[$0] }
        let oddSamples  = stride(from: 1, to: sampleCount, by: 2).map { signal[$0] }

        // ── CONQUER ─────────────────────────────────────────────────────────
        // Recursively transform each half — identical structure to Quicksort's
        // recursive calls on left and right partitions.
        let evenResult = transform(evenSamples)
        let oddResult  = transform(oddSamples)

        // ── COMBINE ─────────────────────────────────────────────────────────
        // Merge the two half-length transforms using "twiddle factors" —
        // complex sine/cosine values that encode the phase relationship between
        // even and odd contributions at each output frequency bin.
        //
        // The twiddle factor for bin index is:
        //   W = cos(−2π · binIndex / sampleCount) + i · sin(−2π · binIndex / sampleCount)
        var result = [_Complex](repeating: _Complex(real: 0, imag: 0), count: sampleCount)
        let halfCount = sampleCount / 2

        for binIndex in 0..<halfCount {
            let angle = -2.0 * Double.pi * Double(binIndex) / Double(sampleCount)
            let twiddle = _Complex(real: Foundation.cos(angle), imag: Foundation.sin(angle))
            let twiddledOdd = twiddle.multiply(oddResult[binIndex])

            // Butterfly operation — the fundamental FFT arithmetic unit
            result[binIndex]             = evenResult[binIndex].add(twiddledOdd)
            result[binIndex + halfCount] = evenResult[binIndex].subtract(twiddledOdd)
        }

        return result
    }

    // MARK: - Real-Input Transform

    /// Computes the forward DFT on real-valued input by wrapping each sample
    /// as a complex value with zero imaginary part.
    ///
    /// - Parameter signal: Real-valued input samples. Length **must** be a power of 2.
    /// - Returns: Complex frequency-domain values.
    /// - Complexity: O(n log n)
    static func transformReal(_ signal: [Double]) -> [_Complex] {
        let complexSignal = signal.map { _Complex(real: $0, imag: 0.0) }
        return transform(complexSignal)
    }

    // MARK: - Inverse Transform

    /// Computes the inverse DFT, reconstructing the original time-domain signal.
    ///
    /// The inverse FFT uses the same butterfly structure as the forward transform.
    /// The relationship between forward and inverse is:
    ///
    ///   IFFT(X)[n] = (1/N) · FFT(conjugate(X))[n]
    ///
    /// Conjugating flips the sign of the imaginary part, which is equivalent to
    /// negating the twiddle factor angles — turning the forward transform's
    /// clockwise rotation into a counter-clockwise one.
    ///
    /// - Parameter spectrum: Complex frequency-domain values from `transform(_:)`.
    /// - Returns: Real-valued time-domain samples scaled by 1/N.
    static func inverseTransform(_ spectrum: [_Complex]) -> [Double] {
        let sampleCount = Double(spectrum.count)

        // Conjugate: negate imaginary parts to reverse rotation direction
        let conjugated = spectrum.map { _Complex(real: $0.real, imag: -$0.imag) }

        // Forward FFT on the full complex conjugated input (not just real parts)
        let transformed = transform(conjugated)

        // Scale by 1/N and return real parts only
        return transformed.map { $0.real / sampleCount }
    }

    // MARK: - Utility

    /// Returns the smallest power of 2 greater than or equal to the given value.
    ///
    /// Use this to determine the correct padding length before calling `transform`.
    ///
    /// - Parameter value: Any positive integer.
    /// - Returns: The next power of 2 ≥ value (e.g., 6 → 8, 8 → 8, 9 → 16).
    static func nextPowerOfTwo(for value: Int) -> Int {
        guard value > 0 else { return 1 }
        var power = 1
        while power < value { power <<= 1 }
        return power
    }

    /// Returns true when the given value is an exact power of 2.
    static func isPowerOfTwo(_ value: Int) -> Bool {
        value > 0 && (value & (value - 1)) == 0
    }
}
