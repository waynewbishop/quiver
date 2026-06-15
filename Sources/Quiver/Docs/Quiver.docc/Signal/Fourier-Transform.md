# Fourier Transform

Decompose a signal into its frequency components using the Fast Fourier Transform.

## Overview

The **Cooley-Tukey** Fast Fourier Transform, published in 1965, is a pillar of computational science. It converts measurements over time into a map of the repeating cycles hidden within data. This algorithm is essential across fields: audio fingerprinting identifies songs, medical devices extract breathing rates from heart signals, radar systems separate moving targets, and JPEG compression discards imperceptible frequencies.

### Signal processing in Quiver

Until now, our work (linear regression, rolling means, and classification models) has operated in the time domain, measuring values at each point to relate them to their neighbors. The Fourier transform operates in the **frequency domain**, revealing the repeating patterns beneath the noise.

The underlying connection to Quiver is direct. The Discrete Fourier Transform is matrix multiplication: for a signal vector `x`, the output `X` is `F · x`, where `F` is a matrix of complex roots of unity. The FFT factors this matrix into sparse sub-matrices, reducing the computational cost from `O(n²)` to `O(n log n)`. The primitives we already use, transposition, element-wise arithmetic, and magnitude, are the building blocks of this algorithm.

### On-device frequency analysis

For iOS and watchOS developers, the Fourier transform answers questions time-domain tools cannot. A rolling mean smooths a noisy heart rate signal but cannot identify the underlying rhythm driving the oscillation. A <doc:Linear-Regression> fits a trend line through sensor data but cannot detect a periodic pattern repeating every four seconds.

We use the Fourier transform directly on-device in milliseconds, transforming the same arrays used for our classifiers and regression models into frequency insights.

### The intersection with machine learning

The Fourier transform is not a machine learning model; it has no `fit`, no `predict`, and no trained state. It is a pure tool: an array goes in, and frequency amplitudes come out. We use it for feature engineering: a dominant frequency from an accelerometer window becomes a new feature column; spectral entropy from a heart rate signal becomes a measure of rhythm irregularity. These Fourier-derived features feed into <doc:Nearest-Neighbors-Classification>, <doc:KMeans-Clustering>, or <doc:Naive-Bayes>, providing crucial information about periodicity that time-domain features miss.

### How it works

Quiver uses the Cooley-Tukey radix-2 algorithm, a divide-and-conquer approach that computes the Discrete Fourier Transform in `O(n log n)` time. It splits the signal into even and odd samples, recurses on each half, and combines the results using complex sine and cosine "twiddle factors" to align them with the correct frequency bins. Like Quicksort, this produces the recurrence `T(n) = 2T(n/2) + O(n)`, though because the split is always exactly half, the `O(n log n)` performance is a guaranteed worst case.

> Note: For a full walkthrough of how this divide-and-conquer recurrence produces `O(n log n)` performance, see [Chapter 5 — Advanced Sorting](https://waynewbishop.github.io/swift-algorithms/05-advanced-sorting.html) in *Swift Algorithms & Data Structures*.

### Preparing the input

The radix-2 algorithm requires a signal length that is a power of two (8, 16, 32, 64, etc.). Our `padded(toPowerOfTwo:)` method handles this by appending zeros to the next valid length:

```swift
import Quiver

let raw = [1.0, 0.5, -0.3, 0.8, 0.2]         // 5 samples
let signal = raw.padded(toPowerOfTwo: 0.0)     // 8 samples, zero-padded
```

Our convenience methods `fourierDominantFrequency(sampleRate:windowed:)` and `fourierSpectrum(sampleRate:windowed:)` manage padding internally.

### Computing the spectrum

`fourierMagnitude` returns the amplitude at each frequency bin, showing the strength of each frequency component. `fourierFrequencies(sampleRate:)` maps bin indices to frequencies in Hz. These arrays are parallel: `frequencies[i]` is the frequency with amplitude `magnitudes[i]`:

```swift
import Quiver

// Generate a 440 Hz tone (concert A) at 8000 Hz sample rate
let sampleRate = 8000.0
let signal = [Double].sineWave(frequency: 440.0, sampleRate: sampleRate, count: 256)

// Compute the frequency spectrum
let magnitudes  = signal.fourierMagnitudeHalf()
let frequencies = signal.fourierFrequenciesHalf(sampleRate: sampleRate)

// Inspect the spectrum as a named-column table
let spectrum = Panel(["frequency": frequencies, "magnitude": magnitudes])
print(spectrum.summary())
```

The <doc:Panel> `summary()` readout provides descriptive statistics for each column, helping us identify whether the magnitude has a single dominant peak or distributed energy.

### Positive-frequency half

For real-valued signals, the upper half of the Fourier output mirrors the lower half and provides no additional information. `fourierMagnitudeHalf` and `fourierFrequenciesHalf(sampleRate:)` return only the positive-frequency half: the range from 0 Hz up to the Nyquist frequency. Use these methods for plotting spectra or extracting features.

### Dominant frequency

`fourierDominantFrequency(sampleRate:)` returns the frequency with the highest amplitude. For most applications (pitch detection, cadence extraction, or cycle identification) this single value is sufficient:

```swift
import Quiver

// One call with automatic padding and windowing
let dominant = sensorData.fourierDominantFrequency(sampleRate: 1000.0, windowed: true)
print("Primary cycle: \(dominant ?? 0) Hz")
```

The convenience overload with `windowed: true` applies a Hann window and pads to a power of two automatically.

### Windowing

Slicing a signal into a finite chunk creates abrupt edges, generating artificial high-frequency content known as spectral leakage. `hannWindowed` tapers the signal smoothly to zero at both ends to suppress this:

```swift
let windowed = signal.hannWindowed()
let magnitudes = windowed.padded(toPowerOfTwo: 0.0).fourierMagnitudeHalf()
```

### Breathing rate from heart rate variability

Respiration modulates the intervals between heartbeats at 0.15 to 0.40 Hz—a phenomenon called respiratory sinus arrhythmia. We can extract the breathing rate from a window of R-R intervals without a dedicated sensor:

```swift
import Quiver

// Simulated R-R intervals — 120 seconds at ~60 BPM
// with respiratory modulation at 0.25 Hz (15 breaths/min)
let sampleRate = 4.0
let rrSignal = [Double].sineWave(
    frequency: 0.25, sampleRate: sampleRate, count: 480,
    amplitude: 50.0, offset: 1000.0
)

// Remove the DC offset — standard practice in HRV analysis,
// as the large baseline (1000ms) would otherwise dominate the spectrum.
let signalMean = rrSignal.mean() ?? 0.0
let centered = rrSignal - signalMean

// Extract the dominant frequency and convert to breaths per minute
let dominant = centered.fourierDominantFrequency(sampleRate: sampleRate, windowed: true)
let breathsPerMinute = (dominant ?? 0) * 60.0  // 15.0
```

### Phase and inverse transform

`fourierPhase` returns the phase angle at each frequency bin: where in its cycle each component sits at time zero. `fourierInverse` performs an inverse transform, treating inputs as complex numbers with zero phase. Because phase information is not preserved in the current public API, a forward-then-inverse round trip will not reconstruct the original signal exactly. This limitation may be addressed in a future release.

### Concurrency

Fourier methods are stateless functions on `[Double]` and do not modify their input, making them implicitly safe to use concurrently. All operations work within `Task` and `async` functions with no extra annotations.

> Experiment: **The Quiver Notebook** is the right place to watch a peak find its frequency. Build a tone with `sineWave(frequency:sampleRate:count:)`, take its `fourierMagnitudeHalf()` against `fourierFrequenciesHalf(sampleRate:)`, and confirm the tallest bin sits at the frequency you chose. Then add a second `sineWave` at a different frequency to the first and re-run — a second peak rises exactly where the new tone lives. Watching two tones resolve into two distinct bins is the fastest way to feel what decomposing a signal into frequencies means. See <doc:Quiver-Notebook>.

## Topics

### Spectrum
- ``Swift/Array/fourierMagnitude()``
- ``Swift/Array/fourierMagnitudeHalf()``
- ``Swift/Array/fourierFrequencies(sampleRate:)``
- ``Swift/Array/fourierFrequenciesHalf(sampleRate:)``
- ``Swift/Array/fourierPhase()``

### Convenience
- ``Swift/Array/fourierDominantFrequency(sampleRate:)``
- ``Swift/Array/fourierDominantFrequency(sampleRate:windowed:)``
- ``Swift/Array/fourierSpectrum(sampleRate:windowed:)``

### Inverse
- ``Swift/Array/fourierInverse()``

### Signal preparation
- ``Swift/Array/padded(toPowerOfTwo:)``
- ``Swift/Array/hannWindowed()``
- ``Swift/Array/sineWave(frequency:sampleRate:count:amplitude:offset:)``

