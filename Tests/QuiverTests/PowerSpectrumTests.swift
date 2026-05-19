import XCTest
@testable import Quiver

/// Pure-Swift tests for `powerSpectralDensity(sampleRate:windowed:)` and `PowerSpectrum`.
///
/// All expected values are analytically derivable from the input signal (peak frequency,
/// Parseval's theorem, band-energy partitioning). No external fixtures.
final class PowerSpectrumTests: XCTestCase {

    // MARK: - Peak detection

    func testSineAt5HzPeaksAt5Hz() throws {
        let psd = try unwrap(pureSine(frequency: 5.0, sampleRate: 100.0, count: 1024)
            .powerSpectralDensity(sampleRate: 100.0))
        // Exact 5 Hz lands on a bin (5 · 1024 / 100 = 51.2 — off-grid, so the peak
        // will be at bin 51 (= 4.98 Hz) or 52 (= 5.08 Hz). Tolerance allows for that.
        XCTAssertEqual(psd.dominantFrequency, 5.0, accuracy: 0.1)
    }

    func testSineAtPowerOfTwoFriendlyFrequencyLandsExactly() throws {
        // 5 cycles in 1024 samples at Fs=100 → frequency = 5 · 100/1024 ≈ 0.488 Hz,
        // but the signal IS exactly that frequency, so the peak should be at bin 5
        // with no leakage.
        let fs = 100.0
        let n = 1024
        let cycles = 5
        let exactHz = Double(cycles) * fs / Double(n)
        let signal = pureSine(frequency: exactHz, sampleRate: fs, count: n)
        let psd = try unwrap(signal.powerSpectralDensity(sampleRate: fs))
        XCTAssertEqual(psd.dominantFrequency, exactHz, accuracy: 1e-9)
    }

    func testTwoToneSpectrumHasPeaksAtBothFrequencies() throws {
        // Signal = sin(2π·5t) + sin(2π·12t), sampled at 100 Hz, N=1024.
        let fs = 100.0
        let n = 1024
        let f1 = 5.0
        let f2 = 12.0
        let signal = (0..<n).map { i -> Double in
            let t = Double(i) / fs
            return Foundation.sin(2 * .pi * f1 * t) + Foundation.sin(2 * .pi * f2 * t)
        }
        let psd = try unwrap(signal.powerSpectralDensity(sampleRate: fs))
        // Energy near each peak should be much larger than energy in a quiet band.
        let energyNear5 = psd.bandEnergy(in: 4.5...5.5)
        let energyNear12 = psd.bandEnergy(in: 11.5...12.5)
        let energyQuiet = psd.bandEnergy(in: 20.0...30.0)
        XCTAssertGreaterThan(energyNear5, 50 * energyQuiet)
        XCTAssertGreaterThan(energyNear12, 50 * energyQuiet)
    }

    // MARK: - Parseval's theorem

    func testParsevalHoldsForPowerOfTwoUnwindowed() throws {
        // SciPy's "density" scaling: P[k] = |X[k]|² / (Fs · N). The discrete
        // Parseval identity in this scaling is Σ P[k] = Σ x²[n] / Fs (sum of
        // P, not sum times Δf — the Fs factor in the density absorbs Δf).
        let fs = 100.0
        let n = 1024
        let signal = pureSine(frequency: 5.0, sampleRate: fs, count: n)
        let psd = try unwrap(signal.powerSpectralDensity(sampleRate: fs))
        let spectralEnergy = psd.densities.reduce(0.0, +)
        let timeEnergy = signal.reduce(0.0) { $0 + $1 * $1 } / fs
        let relativeError = abs(spectralEnergy - timeEnergy) / timeEnergy
        XCTAssertLessThan(relativeError, 1e-10,
            "Parseval relative error: \(relativeError)")
    }

    func testParsevalApproximatelyHoldsAfterZeroPadding() throws {
        // Zero-padding adds bins but no signal energy; Parseval still holds
        // because the padded zeros contribute zero squared.
        let fs = 100.0
        let n = 1000  // pads to 1024
        let signal = pureSine(frequency: 5.0, sampleRate: fs, count: n)
        let psd = try unwrap(signal.powerSpectralDensity(sampleRate: fs))
        let spectralEnergy = psd.densities.reduce(0.0, +)
        let timeEnergy = signal.reduce(0.0) { $0 + $1 * $1 } / fs
        let relativeError = abs(spectralEnergy - timeEnergy) / timeEnergy
        XCTAssertLessThan(relativeError, 1e-10,
            "Parseval (padded) relative error: \(relativeError)")
    }

    // MARK: - Zero-padding behaviour

    func testPaddedLengthIsNextPowerOfTwo() throws {
        // 1000 samples → pads to 1024 → one-sided length = 513.
        let signal = pureSine(frequency: 5.0, sampleRate: 100.0, count: 1000)
        let psd = try unwrap(signal.powerSpectralDensity(sampleRate: 100.0))
        XCTAssertEqual(psd.frequencies.count, 513)
        XCTAssertEqual(psd.densities.count, 513)
    }

    func testFrequencyAxisSpansZeroToNyquist() throws {
        let fs = 100.0
        let signal = pureSine(frequency: 5.0, sampleRate: fs, count: 1024)
        let psd = try unwrap(signal.powerSpectralDensity(sampleRate: fs))
        let first = try XCTUnwrap(psd.frequencies.first)
        let last = try XCTUnwrap(psd.frequencies.last)
        XCTAssertEqual(first, 0.0, accuracy: 1e-12)
        XCTAssertEqual(last, fs / 2, accuracy: 1e-12)
    }

    func testFrequencyAxisIsEvenlySpaced() throws {
        let fs = 100.0
        let signal = pureSine(frequency: 5.0, sampleRate: fs, count: 1024)
        let psd = try unwrap(signal.powerSpectralDensity(sampleRate: fs))
        let deltaF = psd.frequencies[1] - psd.frequencies[0]
        for i in 1..<psd.frequencies.count {
            XCTAssertEqual(psd.frequencies[i] - psd.frequencies[i - 1], deltaF, accuracy: 1e-12)
        }
    }

    // MARK: - bandEnergy

    func testBandEnergyAroundPeakDominatesQuietBand() throws {
        let psd = try unwrap(pureSine(frequency: 5.0, sampleRate: 100.0, count: 1024)
            .powerSpectralDensity(sampleRate: 100.0))
        let nearPeak = psd.bandEnergy(in: 4.0...6.0)
        let farFromPeak = psd.bandEnergy(in: 30.0...40.0)
        XCTAssertGreaterThan(nearPeak, 1000 * farFromPeak)
    }

    func testBandEnergyZeroForBandWithNoBins() throws {
        let psd = try unwrap(pureSine(frequency: 5.0, sampleRate: 100.0, count: 1024)
            .powerSpectralDensity(sampleRate: 100.0))
        // A band entirely above Nyquist contains no bins → zero energy.
        XCTAssertEqual(psd.bandEnergy(in: 100.0...200.0), 0.0, accuracy: 1e-12)
    }

    func testBandEnergyPartitionsTotalEnergy() throws {
        // bandEnergy is Σ density · Δf over bins in the range. Summing across
        // [0, Nyquist] should recover Σ density · Δf for the full one-sided spectrum.
        let psd = try unwrap(pureSine(frequency: 5.0, sampleRate: 100.0, count: 1024)
            .powerSpectralDensity(sampleRate: 100.0))
        let fullBand = psd.bandEnergy(in: 0.0...50.0)
        let deltaF = psd.frequencies[1] - psd.frequencies[0]
        let totalBandEnergy = psd.densities.reduce(0.0, +) * deltaF
        XCTAssertEqual(fullBand, totalBandEnergy, accuracy: 1e-10)
    }

    // MARK: - Windowing

    func testWindowedSpectrumStillPeaksAtSineFrequency() throws {
        let psd = try unwrap(pureSine(frequency: 5.0, sampleRate: 100.0, count: 1024)
            .powerSpectralDensity(sampleRate: 100.0, windowed: true))
        XCTAssertEqual(psd.dominantFrequency, 5.0, accuracy: 0.2)
    }

    func testWindowingReducesSpectralLeakage() throws {
        // For an off-grid frequency, the unwindowed PSD smears energy into adjacent
        // bins; Hann windowing concentrates it. Compare leakage at a distant band.
        let fs = 100.0
        let n = 1024
        // 5.3 Hz is off-grid (bin = 5.3 · 1024/100 = 54.272 — not an integer).
        let signal = pureSine(frequency: 5.3, sampleRate: fs, count: n)
        let unwindowed = try unwrap(signal.powerSpectralDensity(sampleRate: fs))
        let windowed = try unwrap(signal.powerSpectralDensity(sampleRate: fs, windowed: true))
        let leakageBand = 15.0...20.0
        let unwindowedLeakage = unwindowed.bandEnergy(in: leakageBand)
        let windowedLeakage = windowed.bandEnergy(in: leakageBand)
        XCTAssertLessThan(windowedLeakage, unwindowedLeakage,
            "Hann windowing should reduce far-band leakage")
    }

    // MARK: - Edge cases

    func testReturnsNilForTooFewSamples() {
        XCTAssertNil([Double]().powerSpectralDensity(sampleRate: 100))
        XCTAssertNil([1.0].powerSpectralDensity(sampleRate: 100))
    }

    func testDominantFrequencyOnEmptySpectrumIsZero() {
        let psd = PowerSpectrum(sampleRate: 100, frequencies: [], densities: [])
        XCTAssertEqual(psd.dominantFrequency, 0.0)
    }

    func testDominantFrequencyOnDCOnlySpectrumIsZero() {
        // A spectrum with only the DC bin has no non-DC peak.
        let psd = PowerSpectrum(sampleRate: 100, frequencies: [0.0], densities: [42.0])
        XCTAssertEqual(psd.dominantFrequency, 0.0)
    }

    // MARK: - Codable / Equatable

    func testCodableRoundTrip() throws {
        let original = try unwrap(pureSine(frequency: 5.0, sampleRate: 100.0, count: 1024)
            .powerSpectralDensity(sampleRate: 100.0))
        let encoded = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(PowerSpectrum.self, from: encoded)
        XCTAssertEqual(original, restored)
    }

    func testDescriptionContainsKeyFacts() throws {
        let psd = try unwrap(pureSine(frequency: 5.0, sampleRate: 100.0, count: 1024)
            .powerSpectralDensity(sampleRate: 100.0))
        let text = psd.description
        XCTAssertTrue(text.contains("PowerSpectrum"))
        XCTAssertTrue(text.contains("dominant"))
        XCTAssertTrue(text.contains("Nyquist"))
        XCTAssertTrue(text.contains("bins"))
    }

    // MARK: - Helpers

    private func pureSine(frequency: Double, sampleRate: Double, count: Int) -> [Double] {
        (0..<count).map { i in
            Foundation.sin(2 * .pi * frequency * Double(i) / sampleRate)
        }
    }

    private func unwrap<T>(_ value: T?, file: StaticString = #file, line: UInt = #line) throws -> T {
        try XCTUnwrap(value, file: file, line: line)
    }
}
