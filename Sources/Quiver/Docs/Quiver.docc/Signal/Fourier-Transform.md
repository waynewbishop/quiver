# Fourier Transform

Decompose a signal into its frequency components using the Fast Fourier Transform.

## Overview

The [Cooley-Tukey](https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm) Fast Fourier Transform, published in 1965, is one of the most widely used algorithms in computational science. It converts a sequence of measurements over time into a map of the repeating cycles hidden inside that data. Audio fingerprinting services use it to identify songs from a few seconds of sound. Medical devices use it to extract breathing rate from a heart rate signal. Radar systems use it to separate moving targets from stationary clutter. JPEG compression uses it to discard frequencies the human eye cannot perceive. The algorithm appears in virtually every field that works with signals, and it has been a standard topic in computer science and electrical engineering curricula for decades.

### Signal processing and Quiver

Everything in Quiver up to this point — [linear regression](<doc:Linear-Regression>), rolling means, [K-Nearest Neighbors](<doc:Nearest-Neighbors-Classification>), [K-Means](<doc:KMeans-Clustering>) — operates in the time domain. These tools measure values at each point and ask how those values relate to their neighbors: trends, clusters, distances, predictions. The Fourier transform operates in the frequency domain. It measures the signal itself and reveals which repeating cycles are hidden inside it. This is a fundamentally different kind of question, and it is the reason signal processing exists as its own discipline.

The connection to Quiver's existing surface is direct. Underneath, the [Discrete Fourier Transform](https://en.wikipedia.org/wiki/Discrete_Fourier_transform) is [matrix multiplication](<doc:Matrix-Operations>): given a signal vector `x`, the output `X` is the product `F · x`, where `F` is a matrix of complex roots of unity. The FFT is the discovery that this particular matrix has enough internal structure to be factored into sparse sub-matrices, reducing the cost from `O(n²)` to `O(n log n)`. The same matrix operations Quiver already provides — transposition, element-wise arithmetic, magnitude — are the primitives the algorithm is built from.

### On-device frequency analysis

For iOS and watchOS developers, the Fourier transform opens a category of questions that time-domain tools cannot answer. A `rollingMean` smooths a noisy heart rate signal but cannot identify which underlying rhythm is driving the oscillation. A [`LinearRegression`](<doc:Linear-Regression>) fits a trend line through sensor data but cannot detect a periodic pattern that repeats every 4 seconds.

The Fourier transform answers these directly: given a stream of accelerometer readings during a run, it identifies the stride frequency. Given a window of R-R intervals at rest, it extracts the breathing rate. Given a vibration signal from an industrial sensor, it isolates the frequency of a failing bearing. These are on-device computations that run in milliseconds on the same `[Double]` arrays that feed Quiver's classifiers and regression models.

### The intersection with machine learning

The Fourier transform is not a machine learning model. There is no `fit` method, no `predict` method, and no trained state. It is a pure mathematical transform: an array goes in, an array of frequency amplitudes comes out. Its value in a machine learning pipeline is as a feature engineering step. A dominant frequency extracted from an accelerometer window becomes a new column in a feature vector. A spectral entropy value computed from a heart rate signal becomes a measure of how irregular the rhythm is. These Fourier-derived features feed into [`KNearestNeighbors`](<doc:Nearest-Neighbors-Classification>), [`KMeans`](<doc:KMeans-Clustering>), or [`GaussianNaiveBayes`](<doc:Naive-Bayes>) as additional inputs, giving the classifier information about periodicity that time-domain features alone cannot provide.

### How it works

Quiver uses the Cooley-Tukey radix-2 algorithm, a divide-and-conquer approach that computes the Discrete Fourier Transform in `O(n log n)` time. The algorithm splits the signal into even-indexed and odd-indexed samples, recurses on each half, and combines the results using complex sine and cosine values called twiddle factors — values that rotate each odd-indexed contribution by the correct angle so it aligns with the corresponding frequency bin in the output. The recursion has the same structure as Quicksort — both algorithms split their input in half at each level and recombine in `O(n)` work per level, producing the recurrence `T(n) = 2T(n/2) + O(n)`. Unlike Quicksort, the FFT always splits exactly in half, so its `O(n log n)` is a guaranteed worst case, not an average.

> Tip: For a full walkthrough of how this divide-and-conquer recurrence produces `O(n log n)` performance, see [Chapter 5 — Advanced Sorting](https://waynewbishop.github.io/swift-algorithms/05-advanced-sorting.html) in *Swift Algorithms & Data Structures*.

### Preparing the input

The radix-2 algorithm requires the signal length to be a power of 2 (8, 16, 32, 64, 128, 256 and so on). The `padded(toPowerOfTwo:)` method handles this by appending zeros to the next valid length:

```swift
import Quiver

let raw = [1.0, 0.5, -0.3, 0.8, 0.2]         // 5 samples
let signal = raw.padded(toPowerOfTwo: 0.0)     // 8 samples, zero-padded
```

The convenience methods `fourierDominantFrequency(sampleRate:windowed:)` and `fourierSpectrum(sampleRate:windowed:)` handle padding internally, so manual padding is only necessary when calling the lower-level methods directly.

### Computing the spectrum

The `fourierMagnitude` method returns the amplitude at each frequency bin, showing how strongly each frequency is present in the signal. The `fourierFrequencies(sampleRate:)` method maps each bin index to its corresponding frequency in Hz. The two arrays are parallel — `frequencies[i]` is the frequency whose amplitude is `magnitudes[i]`:

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

The `summary()` readout shows the mean, standard deviation, min, max, and quartiles for each column — useful for spotting whether the magnitude column has a single dominant peak (a high max relative to the mean) or distributed energy across many bins.

> Tip: Try this in the <doc:Quiver-Notebook>. Paste the snippet, change the `frequency` argument to `sineWave` from 440 Hz to a different value, and re-run — the peak in `magnitudes` moves to match. Watching the peak shift in real time is the fastest way to build intuition for what the Fourier transform is doing.

### Positive-frequency half

For real-valued signals, the upper half of the Fourier output is a mirror of the lower half and carries no additional information. The `fourierMagnitudeHalf` and `fourierFrequenciesHalf(sampleRate:)` methods return only the positive-frequency half — the range from 0 Hz up to the Nyquist frequency. These are the methods to use when plotting a spectrum or extracting features.

> Tip: The `Half` variants return `count / 2` elements. Both arrays are indexed identically, so they can be passed directly to Swift Charts as parallel x and y values.

### Dominant frequency

The `fourierDominantFrequency(sampleRate:)` method returns the single frequency with the highest amplitude. For most applications — pitch detection, cadence extraction, cycle identification in time series — this is the only Fourier output needed:

```swift
import Quiver

// One call with automatic padding and windowing
let dominant = sensorData.fourierDominantFrequency(sampleRate: 1000.0, windowed: true)
print("Primary cycle: \(dominant ?? 0) Hz")
```

The convenience overload with the `windowed:` parameter applies a Hann window and pads to a power of two internally, so the caller passes raw signal data without any preparation.

### Windowing

When a signal is sliced into a finite chunk, abrupt edges at the start and end create artificial high-frequency content called spectral leakage. The `hannWindowed` method applies the most widely used window function, tapering the signal smoothly to zero at both ends to suppress this artifact:

```swift
let windowed = signal.hannWindowed()
let magnitudes = windowed.padded(toPowerOfTwo: 0.0).fourierMagnitudeHalf()
```

> Tip: The `fourierSpectrum(sampleRate:windowed:)` method combines windowing, padding, and spectrum computation into a single call that returns paired `(frequency, magnitude)` tuples ready for charting.

### Breathing rate from heart rate variability

Respiration modulates the intervals between heartbeats at a frequency between 0.15 and 0.40 Hz, a well-established phenomenon called respiratory sinus arrhythmia. The Fourier transform extracts the dominant respiratory frequency from a window of R-R interval data, recovering the breathing rate without a dedicated respiratory sensor:

```swift
import Quiver

// Simulated R-R intervals at rest — 120 seconds at ~60 BPM
// with respiratory modulation at 0.25 Hz (15 breaths/min)
let sampleRate = 4.0
let rrSignal = [Double].sineWave(
    frequency: 0.25, sampleRate: sampleRate, count: 480,
    amplitude: 50.0, offset: 1000.0
)

// Remove the DC offset before transforming — standard practice in
// frequency-domain HRV analysis, because the large baseline (1000ms)
// would otherwise dominate the spectrum
let signalMean = rrSignal.mean() ?? 0.0
let centered = rrSignal - signalMean

// Extract the dominant frequency and convert to breaths per minute
let dominant = centered.fourierDominantFrequency(sampleRate: sampleRate, windowed: true)
let breathsPerMinute = (dominant ?? 0) * 60.0  // 15.0
```

This technique is common in wearable health devices where direct respiratory measurement is impractical. For end-to-end patterns covering session lifecycle, sensor streams, and on-device inference, see <doc:watchOS-Guide>.

### Phase and inverse transform

The `fourierPhase` method returns the phase angle at each frequency bin — where in its cycle each component sits at time zero. Most applications use magnitude only. Phase analysis is relevant in communications, audio engineering, and interference detection.

The `fourierInverse` method performs an inverse transform on real-valued input, treating each element as a complex number with zero phase. Because phase information is not preserved in the public API, a forward-then-inverse round trip does not reconstruct the original signal exactly. For applications that require exact reconstruction — such as signal filtering, where unwanted frequency bins are zeroed out before reconstructing — the full complex spectrum (magnitude and phase) is needed. This is a limitation of the current public API surface and may be expanded in a future release.

### Concurrency

Fourier methods are stateless functions on `[Double]` that return new arrays without modifying the input. There is no type to conform to `Sendable` because the methods operate on Swift's `Array`, which is already a value type safe to pass between tasks. All Fourier operations work inside `Task`, `Task.detached`, and `async` functions with no additional annotations. For the full set of concurrency patterns, see <doc:Concurrency-Primer>.

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

