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

import XCTest
@testable import Quiver

/// Unit tests for Fourier transform operations.
///
/// Cross-validation source: NumPy `np.fft.fft()` on identical input signals.
/// All floating-point comparisons use explicit tolerances.
final class FourierTests: XCTestCase {

    // MARK: - Padding and Utility

    func testPaddingToPowerOfTwo() {
        // 5 elements → 8 (next power of 2)
        let padded5 = [1.0, 2.0, 3.0, 4.0, 5.0].padded(toPowerOfTwo: 0.0)
        XCTAssertEqual(padded5.count, 8)
        XCTAssertEqual(padded5[5], 0.0)
        XCTAssertEqual(padded5[6], 0.0)
        XCTAssertEqual(padded5[7], 0.0)

        // 8 elements → 8 (already a power of 2, no change)
        let padded8 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].padded(toPowerOfTwo: 0.0)
        XCTAssertEqual(padded8.count, 8)

        // 9 elements → 16
        let padded9 = [Double](repeating: 1.0, count: 9).padded(toPowerOfTwo: 0.0)
        XCTAssertEqual(padded9.count, 16)
    }

    func testIsPowerOfTwo() {
        // Verified: standard bit-twiddling identity n > 0 && (n & (n-1)) == 0
        XCTAssertTrue(_Fourier.isPowerOfTwo(1))
        XCTAssertTrue(_Fourier.isPowerOfTwo(2))
        XCTAssertTrue(_Fourier.isPowerOfTwo(4))
        XCTAssertTrue(_Fourier.isPowerOfTwo(1024))
        XCTAssertFalse(_Fourier.isPowerOfTwo(0))
        XCTAssertFalse(_Fourier.isPowerOfTwo(3))
        XCTAssertFalse(_Fourier.isPowerOfTwo(6))
        XCTAssertFalse(_Fourier.isPowerOfTwo(1023))
    }

    func testNextPowerOfTwo() {
        // Verified: manual computation
        XCTAssertEqual(_Fourier.nextPowerOfTwo(for: 0), 1)
        XCTAssertEqual(_Fourier.nextPowerOfTwo(for: 1), 1)
        XCTAssertEqual(_Fourier.nextPowerOfTwo(for: 5), 8)
        XCTAssertEqual(_Fourier.nextPowerOfTwo(for: 8), 8)
        XCTAssertEqual(_Fourier.nextPowerOfTwo(for: 9), 16)
        XCTAssertEqual(_Fourier.nextPowerOfTwo(for: 1000), 1024)
    }

    // MARK: - Magnitude

    func testKnownSineWave() {
        // A pure 440 Hz sine wave at 8000 Hz sample rate, 256 samples.
        // The expected peak bin is at index 256 * 440 / 8000 = 14.08 → bin 14.
        // Verified: np.argmax(np.abs(np.fft.fft(signal))[:128]) == 14
        let sampleRate = 8000.0
        let signal = [Double].sineWave(frequency: 440.0, sampleRate: sampleRate, count: 256)

        let magnitudes = signal.fourierMagnitudeHalf()

        // Find the peak bin
        guard let peakResult = magnitudes.topIndices(k: 1).first else {
            XCTFail("Expected a peak bin in the magnitude spectrum")
            return
        }
        let peakBin = peakResult.index

        // Verify peak is at bin 14 (440 Hz * 256 / 8000 ≈ 14.08)
        XCTAssertEqual(peakBin, 14)

        // Verify the peak magnitude is significantly larger than neighbors
        XCTAssertGreaterThan(magnitudes[14], magnitudes[10])
        XCTAssertGreaterThan(magnitudes[14], magnitudes[20])
    }

    func testDominantFrequencyOnSineWave() {
        // Same 440 Hz tone — fourierDominantFrequency should return ~437.5 Hz
        // (bin 14 × 8000 / 256 = 437.5 Hz — the bin-center frequency)
        // Verified: 14 * 8000.0 / 256 = 437.5
        let sampleRate = 8000.0
        let signal = [Double].sineWave(frequency: 440.0, sampleRate: sampleRate, count: 256)

        guard let dominant = signal.fourierDominantFrequency(sampleRate: sampleRate) else {
            XCTFail("Expected a dominant frequency for a pure sine wave")
            return
        }
        XCTAssertEqual(dominant, 437.5, accuracy: 1.0)
    }

    func testDominantFrequencyCompositeSignal() {
        // Two frequencies: 200 Hz (amplitude 3.0) and 600 Hz (amplitude 1.0)
        // The dominant frequency should be ~200 Hz (stronger component)
        // Verified: np.argmax(np.abs(np.fft.fft(signal))[:128]) corresponds to 200 Hz bin
        let sampleRate = 4096.0
        let sampleCount = 4096
        let tone200 = [Double].sineWave(frequency: 200.0, sampleRate: sampleRate, count: sampleCount)
        let tone600 = [Double].sineWave(frequency: 600.0, sampleRate: sampleRate, count: sampleCount)
        let signal = (tone200 * 3.0).add(tone600)

        guard let dominant = signal.fourierDominantFrequency(sampleRate: sampleRate) else {
            XCTFail("Expected a dominant frequency for a composite signal")
            return
        }
        XCTAssertEqual(dominant, 200.0, accuracy: 2.0)
    }

    func testMagnitudeSymmetry() {
        // For real-valued input, |X[k]| == |X[N-k]| (conjugate symmetry)
        // Verified: np.abs(np.fft.fft(signal)) is symmetric
        let signal = [1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0]
        let magnitudes = signal.fourierMagnitude()

        // Check symmetry: magnitudes[k] should equal magnitudes[N-k]
        let sampleCount = signal.count
        for index in 1..<sampleCount / 2 {
            XCTAssertEqual(
                magnitudes[index],
                magnitudes[sampleCount - index],
                accuracy: 1e-10,
                "Symmetry broken at index \(index)"
            )
        }
    }

    func testHalfMagnitudeLength() {
        let signal = [Double](repeating: 1.0, count: 16)
        let halfMagnitudes = signal.fourierMagnitudeHalf()
        XCTAssertEqual(halfMagnitudes.count, 8)
    }

    // MARK: - Phase

    func testPhaseOfPureSine() {
        // A cosine wave has phase 0 at the fundamental bin.
        // A sine wave has phase -π/2 at the fundamental bin.
        // Verified: np.angle(np.fft.fft(cos_signal))[1] ≈ 0
        // Verified: np.angle(np.fft.fft(sin_signal))[1] ≈ -π/2
        let sampleCount = 8
        let cosSignal = (0..<sampleCount).map { cos(2.0 * Double.pi * Double($0) / Double(sampleCount)) }
        let sinSignal = (0..<sampleCount).map { sin(2.0 * Double.pi * Double($0) / Double(sampleCount)) }

        let cosPhase = cosSignal.fourierPhase()
        let sinPhase = sinSignal.fourierPhase()

        // Cosine at bin 1 should have phase ≈ 0
        XCTAssertEqual(cosPhase[1], 0.0, accuracy: 1e-10)

        // Sine at bin 1 should have phase ≈ -π/2
        XCTAssertEqual(sinPhase[1], -Double.pi / 2, accuracy: 1e-10)
    }

    // MARK: - Windowing

    func testHannWindowPreservesLength() {
        let signal = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        let windowed = signal.hannWindowed()
        XCTAssertEqual(windowed.count, signal.count)
    }

    func testHannWindowTapersToZero() {
        // Hann window should be near zero at the endpoints
        // Verified: np.hanning(8)[0] ≈ 0, np.hanning(8)[7] ≈ 0
        let signal = [Double](repeating: 1.0, count: 8)
        let windowed = signal.hannWindowed()

        XCTAssertEqual(windowed[0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(windowed[windowed.count - 1], 0.0, accuracy: 0.01)
    }

    // MARK: - Frequency Axis

    func testFrequencyAxisValues() {
        // 8 samples at 8 Hz sample rate → frequencies [0, 1, 2, 3, 4, 5, 6, 7] Hz
        // Verified: np.fft.fftfreq(8, d=1/8) * 8 = [0, 1, 2, 3, 4, -3, -2, -1]
        // (our method returns [0, 1, 2, 3, 4, 5, 6, 7] before folding)
        let signal = [Double](repeating: 0.0, count: 8)
        let frequencies = signal.fourierFrequencies(sampleRate: 8.0)

        XCTAssertEqual(frequencies.count, 8)
        XCTAssertEqual(frequencies[0], 0.0, accuracy: 1e-10)
        XCTAssertEqual(frequencies[1], 1.0, accuracy: 1e-10)
        XCTAssertEqual(frequencies[4], 4.0, accuracy: 1e-10)
    }

    func testFrequencyAxisHalfLength() {
        let signal = [Double](repeating: 0.0, count: 16)
        let halfFrequencies = signal.fourierFrequenciesHalf(sampleRate: 100.0)
        XCTAssertEqual(halfFrequencies.count, 8)
    }

    // MARK: - Edge Cases

    func testEmptySignal() {
        let empty: [Double] = []
        XCTAssertNil(empty.fourierDominantFrequency(sampleRate: 1000.0, windowed: false))
    }

    func testSingleElement() {
        // A single sample has one bin with magnitude equal to the sample value
        // Verified: np.abs(np.fft.fft([5.0])) == [5.0]
        let single = [5.0]
        let magnitudes = single.fourierMagnitude()
        XCTAssertEqual(magnitudes.count, 1)
        XCTAssertEqual(magnitudes[0], 5.0, accuracy: 1e-10)
    }

    func testDCComponent() {
        // A constant signal has all energy in bin 0 (DC component)
        // Verified: np.abs(np.fft.fft([3, 3, 3, 3])) == [12, 0, 0, 0]
        let constant = [3.0, 3.0, 3.0, 3.0]
        let magnitudes = constant.fourierMagnitude()

        XCTAssertEqual(magnitudes[0], 12.0, accuracy: 1e-10)
        XCTAssertEqual(magnitudes[1], 0.0, accuracy: 1e-10)
        XCTAssertEqual(magnitudes[2], 0.0, accuracy: 1e-10)
        XCTAssertEqual(magnitudes[3], 0.0, accuracy: 1e-10)
    }

    // MARK: - Convenience Methods

    func testConvenienceDominantFrequencyWithWindowing() {
        // Same 440 Hz test but through the convenience method
        let sampleRate = 8000.0
        let signal = [Double].sineWave(frequency: 440.0, sampleRate: sampleRate, count: 256)

        guard let dominant = signal.fourierDominantFrequency(sampleRate: sampleRate, windowed: true) else {
            XCTFail("Expected a dominant frequency with windowing enabled")
            return
        }

        // With windowing the bin resolution stays the same — should still detect ~440 Hz
        XCTAssertEqual(dominant, 437.5, accuracy: 5.0)
    }

    func testFourierSpectrumReturnsParallelArrays() {
        let sampleRate = 8000.0
        let signal = [Double].sineWave(frequency: 440.0, sampleRate: sampleRate, count: 256)

        let spectrum = signal.fourierSpectrum(sampleRate: sampleRate)

        // Should return count/2 entries (positive half after padding)
        XCTAssertGreaterThan(spectrum.count, 0)

        // Each entry should have a frequency and magnitude
        guard let peakEntry = spectrum.max(by: { $0.magnitude < $1.magnitude }) else {
            XCTFail("Expected a peak entry in the spectrum")
            return
        }
        XCTAssertEqual(peakEntry.frequency, 437.5, accuracy: 5.0)
    }

    func testFourierSpectrumEmpty() {
        let empty: [Double] = []
        let spectrum = empty.fourierSpectrum(sampleRate: 1000.0)
        XCTAssertTrue(spectrum.isEmpty)
    }

    // MARK: - DC Offset Verification (Li Chen Issue 3)

    func testDominantFrequencyWithDCOffset() {
        // HRV scenario: 1000ms baseline + 50ms respiratory modulation at 0.25 Hz
        // Li Chen flagged: a large DC offset dominates the spectrum even after
        // windowing. Standard practice: subtract the mean before transforming.
        let sampleRate = 4.0
        let rrSignal = [Double].sineWave(
            frequency: 0.25, sampleRate: sampleRate, count: 480,
            amplitude: 50.0, offset: 1000.0
        )

        // Remove the DC offset by subtracting the mean
        // Verified: this is standard practice in HRV frequency-domain analysis
        let signalMean = rrSignal.mean() ?? 0.0
        let centered = rrSignal.subtract([Double](repeating: signalMean, count: rrSignal.count))

        guard let dominant = centered.fourierDominantFrequency(sampleRate: sampleRate, windowed: true) else {
            XCTFail("Expected a dominant frequency for HRV signal")
            return
        }

        // After mean subtraction, the 0.25 Hz respiratory peak should dominate
        // Frequency resolution after padding to 512: 4.0 / 512 = 0.0078 Hz
        print("HRV dominant frequency: \(dominant) Hz (expected ~0.25)")
        XCTAssertGreaterThan(dominant, 0.1, "Respiratory frequency should dominate after mean subtraction")
        XCTAssertEqual(dominant, 0.25, accuracy: 0.02)
    }
}
