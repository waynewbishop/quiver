// Copyright 2025 Wayne W Bishop. All rights reserved.
//
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

/// Stress tests for Quiver's performance-critical code paths.
///
/// Input sizes are calibrated to run the full suite in under 5 seconds while
/// still exercising O(n²) and O(n³) code paths meaningfully.
///
/// Run with: `swift test --filter QuiverStressTests`
/// Skip during normal development: `swift test --skip QuiverStressTests`
final class QuiverStressTests: XCTestCase {

    // MARK: - Helpers

    /// Measures execution time and peak memory, printing results to console.
    private func benchmark(_ label: String, _ block: () -> Void) {
        let startMemory = currentMemoryMB()
        let start = CFAbsoluteTimeGetCurrent()
        block()
        let elapsed = CFAbsoluteTimeGetCurrent() - start
        let endMemory = currentMemoryMB()
        let memoryDelta = endMemory - startMemory

        print("""
        ┌─ \(label)
        │  Time:   \(String(format: "%.3f", elapsed))s
        │  Memory: \(String(format: "%.1f", endMemory))MB (Δ \(String(format: "%+.1f", memoryDelta))MB)
        └─ \(elapsed < 1 ? "✓ fast" : elapsed < 5 ? "● moderate" : "▲ slow")
        """)
    }

    /// Returns current memory usage in megabytes.
    private func currentMemoryMB() -> Double {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size) / 4
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? Double(info.resident_size) / 1_048_576.0 : 0
    }

    // MARK: - Matrix Operations (O(n³))

    func testMatrixMultiply() {
        let a = [Double].random(100, 100)
        let b = [Double].random(100, 100)
        benchmark("Matrix Multiply 100×100") {
            let _ = a.multiplyMatrix(b)
        }
    }

    func testMatrixInverse() throws {
        var matrix = [Double].random(100, 100)
        for i in 0..<100 { matrix[i][i] += 100.0 }
        benchmark("Matrix Inverse 100×100") {
            let _ = try? matrix.inverted()
        }
    }

    func testDeterminant() {
        let matrix = [Double].random(150, 150)
        benchmark("Determinant 150×150") {
            let _ = matrix.determinant
        }
    }

    func testLogDeterminant() {
        let matrix = [Double].random(150, 150)
        benchmark("Log Determinant 150×150") {
            let _ = matrix.logDeterminant
        }
    }

    func testTranspose() {
        let matrix = [Double].random(500, 500)
        benchmark("Transpose 500×500") {
            let _ = matrix.transposed()
        }
    }

    // MARK: - K-Nearest Neighbors (O(q·m·d))

    func testKNNEuclidean() {
        let train = [Double].random(1_000, 10)
        let labels = (0..<1_000).map { _ in Int.random(in: 0..<5) }
        let queries = [Double].random(100, 10)
        let model = KNearestNeighbors.fit(features: train, labels: labels, k: 5, metric: .euclidean)
        benchmark("KNN Euclidean 1k train, 100 query") {
            let _ = model.predict(queries)
        }
    }

    func testKNNCosine() {
        let train = [Double].random(1_000, 10)
        let labels = (0..<1_000).map { _ in Int.random(in: 0..<5) }
        let queries = [Double].random(100, 10)
        let model = KNearestNeighbors.fit(features: train, labels: labels, k: 5, metric: .cosine)
        benchmark("KNN Cosine 1k train, 100 query") {
            let _ = model.predict(queries)
        }
    }

    // MARK: - K-Means Clustering (O(iter·n·k·d))

    func testKMeans() {
        let data = [Double].random(1_000, 10)
        benchmark("KMeans 1k samples, k=5") {
            let _ = KMeans.fit(data: data, k: 5, maxIterations: 50, seed: 42)
        }
    }

    // MARK: - Linear Regression (O(n·p² + p³))

    func testLinearRegression() {
        let features = [Double].random(5_000, 10)
        let coefficients = [Double].random(10)
        let targets = features.map { $0.dot(coefficients) + Double.random(in: -0.1...0.1) }
        benchmark("Linear Regression 5k samples, p=10") {
            let _ = try? LinearRegression.fit(features: features, targets: targets)
        }
    }

    // MARK: - Pairwise Operations (O(n²·d))

    func testFindDuplicates() {
        let vectors = [Double].random(500, 20)
        benchmark("findDuplicates 500 vectors") {
            let _ = vectors.findDuplicates(threshold: 0.99)
        }
    }

    func testClusterCohesion() {
        let vectors = [Double].random(500, 20)
        benchmark("clusterCohesion 500 vectors") {
            let _ = vectors.clusterCohesion()
        }
    }

    // MARK: - Gaussian Naive Bayes (O(c·n·d))

    func testNaiveBayes() {
        let features = [Double].random(10_000, 20)
        let labels = (0..<10_000).map { _ in Int.random(in: 0..<5) }
        let queries = [Double].random(1_000, 20)
        benchmark("Naive Bayes fit 10k + predict 1k") {
            let model = GaussianNaiveBayes.fit(features: features, labels: labels)
            let _ = model.predict(queries)
        }
    }

    // MARK: - Fourier Transform (O(n log n))

    func testFourierTransform() {
        // 16,384 samples — realistic audio-length signal
        let signal = [Double].sineWave(frequency: 440.0, sampleRate: 8000.0, count: 16_384)
        benchmark("Fourier Transform 16k samples") {
            let _ = signal.fourierMagnitude()
        }
    }

    func testFourierWithWindowing() {
        // Full pipeline: window → pad → magnitude
        let signal = [Double].sineWave(frequency: 440.0, sampleRate: 8000.0, count: 16_384)
        benchmark("Fourier Windowed Pipeline 16k samples") {
            let _ = signal.hannWindowed().fourierMagnitudeHalf()
        }
    }
}
