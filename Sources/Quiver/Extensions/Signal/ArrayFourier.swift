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

// MARK: - Public API

/// Fourier transform operations on real-valued signal arrays.
///
/// ## Overview
///
/// The Fourier transform converts a time-domain signal — a sequence of measurements
/// over time — into its frequency-domain representation, revealing which cycles are
/// hidden inside the data. This is the algorithm that powers audio fingerprinting,
/// speech recognition, sensor anomaly detection, and image compression.
///
/// ## Algorithm
///
/// Quiver uses the Cooley-Tukey radix-2 FFT, the same divide-and-conquer strategy
/// as Quicksort applied to frequency analysis. The recurrence
/// `T(n) = 2T(n/2) + O(n)` is identical for both algorithms, giving `O(n log n)`.
///
/// ## Input requirements
///
/// The signal length must be a power of 2 (8, 16, 32, 64, 128, 256 …).
/// Use `padded(toPowerOfTwo:)` to prepare a signal of arbitrary length,
/// or use the convenience methods which handle padding automatically.
///
/// ## Real-world example — detecting a musical note
///
/// ```swift
/// import Quiver
///
/// // Generate a 440 Hz tone (concert A) at 8000 Hz sample rate
/// let sampleRate = 8000.0
/// let signal = [Double].sineWave(frequency: 440.0, sampleRate: sampleRate, count: 256)
///
/// // Find the dominant frequency — one call with automatic padding
/// if let dominantFrequency = signal.fourierDominantFrequency(sampleRate: sampleRate, windowed: true) {
///     print("Dominant frequency: \(dominantFrequency) Hz")  // ~440 Hz
/// }
/// ```
public extension Array where Element == Double {

    // MARK: - Core Outputs

    /// Returns the amplitude at each frequency bin.
    ///
    /// The amplitude spectrum shows how strongly each frequency is present in the
    /// signal. A pure 440 Hz tone produces one large spike at the 440 Hz bin and
    /// near-zero values everywhere else.
    ///
    /// The output has the same length as the input. For real-valued signals only the
    /// first `count / 2` bins carry unique information — the second half is a mirror
    /// image. Use `fourierMagnitudeHalf()` to return only the meaningful half.
    ///
    /// - Precondition: `count` must be a power of 2. Use `padded(toPowerOfTwo:)` first
    ///   if the signal length is not already a power of 2.
    /// - Complexity: O(n log n)
    /// - Returns: Amplitude values in the same order as `fourierFrequencies(sampleRate:)`.
    func fourierMagnitude() -> [Double] {
        precondition(
            _Fourier.isPowerOfTwo(self.count),
            "fourierMagnitude() requires a power-of-two length. Use padded(toPowerOfTwo:) first."
        )
        return _Fourier.transformReal(self).map { $0.magnitude }
    }

    /// Returns the amplitude spectrum for only the positive-frequency half.
    ///
    /// For real-valued input, the upper half of the Fourier output is a mirror of
    /// the lower half and carries no additional information. This method returns only
    /// the first `count / 2` bins — the range from DC (0 Hz) up to the Nyquist
    /// frequency (sampleRate / 2).
    ///
    /// Use this when plotting a spectrum — it avoids the confusing mirror image that
    /// appears in the full output.
    ///
    /// - Precondition: `count` must be a power of 2.
    /// - Complexity: O(n log n)
    func fourierMagnitudeHalf() -> [Double] {
        precondition(
            _Fourier.isPowerOfTwo(self.count),
            "fourierMagnitudeHalf() requires a power-of-two length. Use padded(toPowerOfTwo:) first."
        )
        let spectrum = _Fourier.transformReal(self)
        return spectrum.prefix(self.count / 2).map { $0.magnitude }
    }

    /// Returns the phase angle (in radians) at each frequency bin.
    ///
    /// Phase describes where in its cycle each frequency component sits at time zero.
    /// Phase analysis is useful in communications, audio engineering, and interference
    /// detection, though most signal processing applications use magnitude only.
    ///
    /// - Precondition: `count` must be a power of 2.
    /// - Complexity: O(n log n)
    /// - Returns: Phase angles in radians in the range (−π, π].
    func fourierPhase() -> [Double] {
        precondition(
            _Fourier.isPowerOfTwo(self.count),
            "fourierPhase() requires a power-of-two length. Use padded(toPowerOfTwo:) first."
        )
        return _Fourier.transformReal(self).map { $0.phase }
    }

    // MARK: - Frequency Axis

    /// Maps each output bin to its corresponding frequency in Hz.
    ///
    /// The Fourier transform produces `n` bins. Bin at position `index` corresponds
    /// to the frequency:
    ///
    ///   frequency[index] = index × sampleRate / sampleCount
    ///
    /// Always call this alongside `fourierMagnitude()` so bin indices become
    /// human-readable Hz values. The two arrays are parallel — `frequencies[index]`
    /// is the frequency whose amplitude is `magnitudes[index]`.
    ///
    /// - Parameter sampleRate: The number of samples per second used when recording
    ///   the signal (e.g. 44100.0 for CD audio, 8000.0 for phone calls).
    /// - Complexity: O(n)
    func fourierFrequencies(sampleRate: Double) -> [Double] {
        let sampleCount = self.count
        return (0..<sampleCount).map { index in Double(index) * sampleRate / Double(sampleCount) }
    }

    /// Returns only the positive-frequency half of the frequency axis.
    ///
    /// Pair this with `fourierMagnitudeHalf()` when plotting a spectrum — both arrays
    /// have length `count / 2` and are indexed identically.
    ///
    /// - Parameter sampleRate: Samples per second of the original signal.
    /// - Complexity: O(n)
    func fourierFrequenciesHalf(sampleRate: Double) -> [Double] {
        let sampleCount = self.count
        return (0..<sampleCount / 2).map { index in Double(index) * sampleRate / Double(sampleCount) }
    }

    // MARK: - Inverse Transform

    /// Performs an inverse Fourier transform on real-valued input.
    ///
    /// Each input element is treated as a complex number with zero imaginary part.
    /// Because phase information from the forward transform is not preserved in
    /// this method's input, a forward-then-inverse round trip does not reconstruct
    /// the original signal exactly. For exact reconstruction, the full complex
    /// spectrum (magnitude and phase) is needed.
    ///
    /// - Precondition: `count` must be a power of 2.
    /// - Complexity: O(n log n)
    func fourierInverse() -> [Double] {
        precondition(
            _Fourier.isPowerOfTwo(self.count),
            "fourierInverse() requires a power-of-two length."
        )
        let spectrum = self.map { _Complex(real: $0, imag: 0.0) }
        return _Fourier.inverseTransform(spectrum)
    }

    // MARK: - Windowing

    /// Applies a Hann window to reduce spectral leakage before computing the
    /// Fourier transform.
    ///
    /// When a signal is sliced into a finite chunk, abrupt edges at the start and
    /// end create artificial high-frequency content called spectral leakage. A window
    /// function tapers the signal smoothly to zero at both ends, suppressing this
    /// artifact.
    ///
    /// The Hann window is the most widely used window in audio analysis. Apply it
    /// before calling `fourierMagnitude()`:
    ///
    /// ```swift
    /// let windowed = signal.hannWindowed()
    /// let magnitudes = windowed.fourierMagnitude()
    /// ```
    ///
    /// - Complexity: O(n) — uses `linspace`, `cos`, and `multiply` from Quiver.
    /// - Returns: A new array of the same length with the Hann window applied.
    func hannWindowed() -> [Double] {
        let sampleCount = self.count

        // Hann window: w[n] = 0.5 · (1 − cos(2π · n / (sampleCount − 1)))
        // Built entirely from existing Quiver API:
        //   linspace → generates the angle sequence
        //   .cos()   → element-wise cosine (ArrayElementwise.swift)
        //   .multiply → element-wise product (ArrayArithmetic.swift)
        let angles = [Double].linspace(start: 0.0, end: 2.0 * Double.pi, count: sampleCount)
        let window = angles.cos().map { 0.5 * (1.0 - $0) }
        return self.multiply(window)
    }

    // MARK: - Padding

    /// Zero-pads the signal to the next power of two, satisfying the Fourier
    /// transform's input requirement.
    ///
    /// The Cooley-Tukey radix-2 algorithm requires the sample count to be a power
    /// of 2. Zero-padding to the next power of 2 is the standard solution for signals
    /// of arbitrary length. Zero-padding also interpolates the spectrum, producing a
    /// smoother visual curve when plotted.
    ///
    /// ```swift
    /// let raw = [1.0, 0.5, -0.3, 0.8, 0.2]         // 5 samples
    /// let padded = raw.padded(toPowerOfTwo: 0.0)     // 8 samples (next power of 2)
    /// let magnitudes = padded.fourierMagnitude()
    /// ```
    ///
    /// - Parameter value: The value to use for padding (default 0.0).
    /// - Complexity: O(n)
    func padded(toPowerOfTwo value: Double = 0.0) -> [Double] {
        let targetLength = _Fourier.nextPowerOfTwo(for: self.count)
        guard targetLength > self.count else { return self }
        return self + [Double](repeating: value, count: targetLength - self.count)
    }

    // MARK: - Dominant Frequency

    /// Returns the frequency with the highest amplitude in the spectrum,
    /// excluding the DC component (0 Hz).
    ///
    /// This is the single most useful Fourier output for many applications — finding
    /// the fundamental pitch of an audio clip, the dominant vibration frequency of a
    /// sensor, or the period of a time series. The DC component (bin 0) is excluded
    /// because it represents the signal's average value, not a repeating cycle.
    ///
    /// - Parameter sampleRate: Samples per second of the original signal.
    /// - Returns: The frequency in Hz of the strongest non-DC component, or `nil` if
    ///   the signal is empty or has fewer than 2 samples.
    /// - Precondition: `count` must be a power of 2.
    /// - Complexity: O(n log n)
    func fourierDominantFrequency(sampleRate: Double) -> Double? {
        guard self.count > 1 else { return nil }

        precondition(
            _Fourier.isPowerOfTwo(self.count),
            "fourierDominantFrequency() requires a power-of-two length. Use padded(toPowerOfTwo:) first."
        )

        // Use only the positive-frequency half, excluding bin 0 (DC component)
        let magnitudes  = fourierMagnitudeHalf()
        let frequencies = fourierFrequenciesHalf(sampleRate: sampleRate)

        // Skip DC (bin 0) — it represents the signal's average, not a cycle
        let nonDCMagnitudes = Array(magnitudes.dropFirst())
        let peak = nonDCMagnitudes.topIndices(k: 1)
        guard let peakBin = peak.first else { return nil }

        // Offset index by 1 to account for the dropped DC bin
        return frequencies[peakBin.index + 1]
    }

    // MARK: - Convenience Methods

    /// Returns the dominant frequency with automatic padding and optional windowing.
    ///
    /// This is the simplest path from a raw signal to a frequency answer. Padding
    /// and windowing are handled internally so the caller does not need to prepare
    /// the signal manually.
    ///
    /// ```swift
    /// // One call — padding and windowing handled automatically
    /// let dominant = sensorData.fourierDominantFrequency(
    ///     sampleRate: 1000.0, windowed: true
    /// )
    /// ```
    ///
    /// - Parameters:
    ///   - sampleRate: Samples per second of the original signal.
    ///   - windowed: Whether to apply a Hann window before transforming.
    ///     Defaults to `false`.
    /// - Returns: The frequency in Hz of the strongest component, or `nil` if the
    ///   signal is empty.
    /// - Complexity: O(n log n)
    func fourierDominantFrequency(sampleRate: Double, windowed: Bool) -> Double? {
        guard !self.isEmpty else { return nil }

        var prepared = windowed ? self.hannWindowed() : self
        prepared = prepared.padded(toPowerOfTwo: 0.0)
        return prepared.fourierDominantFrequency(sampleRate: sampleRate)
    }

    /// Returns the positive-frequency spectrum as paired frequency and magnitude values,
    /// with automatic padding and optional windowing.
    ///
    /// This is the simplest path from a raw signal to a plottable spectrum. Each
    /// element in the result pairs a frequency in Hz with its corresponding amplitude.
    /// Feed the result directly to Swift Charts or any visualization layer.
    ///
    /// ```swift
    /// let spectrum = signal.fourierSpectrum(sampleRate: 44100.0, windowed: true)
    ///
    /// // Each entry is ready for charting
    /// for bin in spectrum {
    ///     print("\(bin.frequency) Hz → amplitude \(bin.magnitude)")
    /// }
    /// ```
    ///
    /// - Parameters:
    ///   - sampleRate: Samples per second of the original signal.
    ///   - windowed: Whether to apply a Hann window before transforming.
    ///     Defaults to `false`.
    /// - Returns: An array of `(frequency: Double, magnitude: Double)` tuples
    ///   covering the positive-frequency half of the spectrum.
    /// - Complexity: O(n log n)
    func fourierSpectrum(sampleRate: Double, windowed: Bool = false) -> [(frequency: Double, magnitude: Double)] {
        guard !self.isEmpty else { return [] }

        var prepared = windowed ? self.hannWindowed() : self
        prepared = prepared.padded(toPowerOfTwo: 0.0)

        let magnitudes = prepared.fourierMagnitudeHalf()
        let frequencies = prepared.fourierFrequenciesHalf(sampleRate: sampleRate)

        return zip(frequencies, magnitudes).map { (frequency: $0, magnitude: $1) }
    }
}
