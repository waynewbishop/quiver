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

/// A one-sided power spectral density estimate, returned by
/// ``Swift/Array/powerSpectralDensity(sampleRate:windowed:)``.
///
/// `PowerSpectrum` pairs each frequency bin with its energy-per-hertz density so the
/// spectrum can be plotted, queried for its dominant frequency, or integrated across
/// a frequency band — all from the same value. The struct stores the sample rate
/// alongside the frequency axis so band-energy calculations have everything they
/// need without a second argument at the call site.
///
/// The density is expressed as energy per hertz, scaled so that integrating the
/// density over the full one-sided band recovers the signal's total power
/// (Parseval's theorem). The spectrum is one-sided — frequencies from zero to the
/// Nyquist limit (half the sample rate) — with interior bins doubled to preserve
/// total energy.
public struct PowerSpectrum: Equatable, Codable, Sendable {

    /// The sample rate of the source signal, in samples per second.
    public let sampleRate: Double

    /// The frequency in hertz for each density bin. Length is `paddedLength / 2 + 1`,
    /// where `paddedLength` is the next power of two at or above the input length.
    /// Values are evenly spaced from 0 to `sampleRate / 2` inclusive.
    public let frequencies: [Double]

    /// The power spectral density at each frequency bin, in units of `(signal²) / Hz`.
    /// Parallel to ``frequencies``.
    public let densities: [Double]

    public init(sampleRate: Double, frequencies: [Double], densities: [Double]) {
        self.sampleRate = sampleRate
        self.frequencies = frequencies
        self.densities = densities
    }

    /// The frequency in hertz of the strongest non-DC component.
    ///
    /// Skips the DC bin (bin 0) because it represents the signal's average rather than
    /// a repeating cycle. Returns 0 if the spectrum is empty or contains only a DC bin.
    public var dominantFrequency: Double {
        guard densities.count > 1 else { return 0 }
        var bestBin = 1
        var bestValue = densities[1]
        for i in 2..<densities.count {
            if densities[i] > bestValue {
                bestValue = densities[i]
                bestBin = i
            }
        }
        return frequencies[bestBin]
    }

    /// Returns the total energy contained in the frequency band defined by `range`.
    ///
    /// Sums `density · Δf` across every bin whose frequency falls inside the closed
    /// range. `Δf` is the spacing between adjacent frequency bins (`sampleRate /
    /// paddedLength`). This is the discrete-spectrum equivalent of integrating the
    /// density over the band — the units come out as `signal²`, the same units as
    /// the area under the original signal's squared waveform.
    ///
    /// Example:
    /// ```swift
    /// // For a runner, the cadence band is roughly 2–3.5 Hz.
    /// let psd = accelMagnitude.powerSpectralDensity(sampleRate: 50, windowed: true)
    /// let cadenceEnergy = psd?.bandEnergy(in: 2.0...3.5) ?? 0
    /// ```
    ///
    /// - Parameter range: A closed frequency range in hertz.
    /// - Returns: The total energy in the band. Zero if the range contains no bins or
    ///   the spectrum is empty.
    /// - Complexity: O(*n*) where *n* is the number of frequency bins.
    public func bandEnergy(in range: ClosedRange<Double>) -> Double {
        guard frequencies.count >= 2 else { return 0 }
        let deltaF = frequencies[1] - frequencies[0]
        var total = 0.0
        for i in 0..<frequencies.count {
            if range.contains(frequencies[i]) {
                total += densities[i] * deltaF
            }
        }
        return total
    }
}

extension PowerSpectrum: CustomStringConvertible {
    public var description: String {
        let domHz = String(format: "%.4f", dominantFrequency)
        let binCount = densities.count
        let nyquist = String(format: "%.2f", sampleRate / 2)
        return "PowerSpectrum(bins: \(binCount), sampleRate: \(sampleRate) Hz, Nyquist: \(nyquist) Hz, dominant: \(domHz) Hz)"
    }
}
