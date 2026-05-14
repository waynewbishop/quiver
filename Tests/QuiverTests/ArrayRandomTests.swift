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

final class ArrayRandomTests: XCTestCase {

    // Covers 1D, 2D, empty, and uniqueness
    func testRandomUniform() {
        let result1D = [Double].random(5)
        XCTAssertEqual(result1D.count, 5)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 1.0)
        }

        let result2D = [Double].random(3, 4)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)
        for row in result2D {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0.0)
                XCTAssertLessThanOrEqual(value, 1.0)
            }
        }

        XCTAssertTrue([Double].random(0).isEmpty)
        XCTAssertNotEqual([Double].random(10), [Double].random(10))
    }

    // MARK: - Custom Range Tests

    // 1D and 2D random doubles in a custom range produce values in bounds
    func testRandomCustomRange() {
        let result1D = [Double].random(100, in: -5.0...5.0)
        XCTAssertEqual(result1D.count, 100)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, -5.0)
            XCTAssertLessThanOrEqual(value, 5.0)
        }

        let result2D = [Double].random(3, 4, in: 10.0...20.0)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)
        for row in result2D {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 10.0)
                XCTAssertLessThanOrEqual(value, 20.0)
            }
        }
    }

    // MARK: - Normal Distribution Tests

    // Covers 1D default, custom params, 2D, and zero std
    func testRandomNormal() {
        let result1D = [Double].randomNormal(1000)
        XCTAssertEqual(result1D.count, 1000)
        XCTAssertEqual(result1D.mean()!, 0.0, accuracy: 0.2)
        XCTAssertEqual(result1D.standardDeviation()!, 1.0, accuracy: 0.2)

        let custom = [Double].randomNormal(1000, mean: 5.0, standardDeviation: 2.0)
        XCTAssertEqual(custom.mean()!, 5.0, accuracy: 0.3)
        XCTAssertEqual(custom.standardDeviation()!, 2.0, accuracy: 0.3)

        let result2D = [Double].randomNormal(3, 4)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)

        let zeroStd = [Double].randomNormal(10, mean: 3.0, standardDeviation: 0.0)
        for value in zeroStd { XCTAssertEqual(value, 3.0) }
    }

    // MARK: - Exponential Distribution Tests

    // 1D exponential samples have correct mean, are non-negative, and respect edge cases
    func testRandomExponential() {
        let result1D = [Double].randomExponential(5000, rate: 2.0)
        XCTAssertEqual(result1D.count, 5000)
        // Theoretical mean is 1 / rate = 0.5
        XCTAssertEqual(result1D.mean()!, 0.5, accuracy: 0.05)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, 0.0)
        }

        // Default rate of 1.0 has mean 1.0
        let defaultRate = [Double].randomExponential(5000)
        XCTAssertEqual(defaultRate.mean()!, 1.0, accuracy: 0.1)

        XCTAssertTrue([Double].randomExponential(0).isEmpty)
    }

    // 2D exponential variant produces a correctly-shaped matrix of non-negative values
    func testRandomExponential2D() {
        let result2D = [Double].randomExponential(4, 6, rate: 1.5)
        XCTAssertEqual(result2D.count, 4)
        for row in result2D {
            XCTAssertEqual(row.count, 6)
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0.0)
            }
        }
    }

    // MARK: - Binomial Distribution Tests

    // 1D binomial samples have correct mean, integer values, are bounded, and respect edge cases
    func testRandomBinomial() {
        let result1D = [Double].randomBinomial(2000, n: 10, p: 0.3)
        XCTAssertEqual(result1D.count, 2000)
        // Theoretical mean is n * p = 3.0
        XCTAssertEqual(result1D.mean()!, 3.0, accuracy: 0.2)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, 0.0)
            XCTAssertLessThanOrEqual(value, 10.0)
            // Whole-number values
            XCTAssertEqual(value.truncatingRemainder(dividingBy: 1.0), 0.0)
        }

        XCTAssertTrue([Double].randomBinomial(0, n: 5, p: 0.5).isEmpty)
    }

    // n == 0 returns all zeros without panicking
    func testRandomBinomialNZero() {
        let result = [Double].randomBinomial(50, n: 0, p: 0.5)
        XCTAssertEqual(result.count, 50)
        for value in result { XCTAssertEqual(value, 0.0) }
    }

    // p == 0 always returns zeros, p == 1 always returns n
    func testRandomBinomialEdgeProbabilities() {
        let zeroProb = [Double].randomBinomial(50, n: 10, p: 0.0)
        for value in zeroProb { XCTAssertEqual(value, 0.0) }

        let oneProb = [Double].randomBinomial(50, n: 10, p: 1.0)
        for value in oneProb { XCTAssertEqual(value, 10.0) }
    }

    // 2D binomial variant produces a correctly-shaped matrix of bounded integer values
    func testRandomBinomial2D() {
        let result2D = [Double].randomBinomial(3, 5, n: 8, p: 0.4)
        XCTAssertEqual(result2D.count, 3)
        for row in result2D {
            XCTAssertEqual(row.count, 5)
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 0.0)
                XCTAssertLessThanOrEqual(value, 8.0)
                XCTAssertEqual(value.truncatingRemainder(dividingBy: 1.0), 0.0)
            }
        }
    }

    // MARK: - Seeded Reproducibility Tests

    // Same seed produces identical uniform output across two calls
    func testRandomUsingSeededReproducible() {
        var a = SeededRandomNumberGenerator(seed: 99)
        var b = SeededRandomNumberGenerator(seed: 99)
        XCTAssertEqual([Double].random(50, using: &a), [Double].random(50, using: &b))

        var c = SeededRandomNumberGenerator(seed: 99)
        var d = SeededRandomNumberGenerator(seed: 99)
        XCTAssertEqual(
            [Double].random(50, in: -10.0...10.0, using: &c),
            [Double].random(50, in: -10.0...10.0, using: &d)
        )
    }

    // Same seed produces identical normal output across two calls, including 2D
    func testRandomNormalUsingSeededReproducible() {
        var a = SeededRandomNumberGenerator(seed: 7)
        var b = SeededRandomNumberGenerator(seed: 7)
        XCTAssertEqual(
            [Double].randomNormal(100, mean: 5.0, standardDeviation: 2.0, using: &a),
            [Double].randomNormal(100, mean: 5.0, standardDeviation: 2.0, using: &b)
        )

        var c = SeededRandomNumberGenerator(seed: 7)
        var d = SeededRandomNumberGenerator(seed: 7)
        XCTAssertEqual(
            [Double].randomNormal(4, 5, using: &c),
            [Double].randomNormal(4, 5, using: &d)
        )
    }

    // Same seed produces identical exponential output across two calls, including 2D
    func testRandomExponentialUsingSeededReproducible() {
        var a = SeededRandomNumberGenerator(seed: 31)
        var b = SeededRandomNumberGenerator(seed: 31)
        XCTAssertEqual(
            [Double].randomExponential(200, rate: 2.0, using: &a),
            [Double].randomExponential(200, rate: 2.0, using: &b)
        )

        var c = SeededRandomNumberGenerator(seed: 31)
        var d = SeededRandomNumberGenerator(seed: 31)
        XCTAssertEqual(
            [Double].randomExponential(3, 4, rate: 1.5, using: &c),
            [Double].randomExponential(3, 4, rate: 1.5, using: &d)
        )
    }

    // Same seed produces identical binomial output across two calls, including 2D
    func testRandomBinomialUsingSeededReproducible() {
        var a = SeededRandomNumberGenerator(seed: 123)
        var b = SeededRandomNumberGenerator(seed: 123)
        XCTAssertEqual(
            [Double].randomBinomial(200, n: 10, p: 0.4, using: &a),
            [Double].randomBinomial(200, n: 10, p: 0.4, using: &b)
        )

        var c = SeededRandomNumberGenerator(seed: 123)
        var d = SeededRandomNumberGenerator(seed: 123)
        XCTAssertEqual(
            [Double].randomBinomial(3, 4, n: 5, p: 0.5, using: &c),
            [Double].randomBinomial(3, 4, n: 5, p: 0.5, using: &d)
        )
    }

    // The using: overload with SystemRandomNumberGenerator behaves like the non-using: form
    // (smoke test: check shape and bounds, not exact values, since the system generator is non-reproducible)
    func testRandomUsingSystemMatchesNonSeeded() {
        var sys = SystemRandomNumberGenerator()
        let viaUsing = [Double].randomNormal(500, mean: 0.0, standardDeviation: 1.0, using: &sys)
        let nonUsing = [Double].randomNormal(500)
        XCTAssertEqual(viaUsing.count, nonUsing.count)
        // Both samples should have means roughly near 0
        XCTAssertEqual(viaUsing.mean()!, 0.0, accuracy: 0.3)
        XCTAssertEqual(nonUsing.mean()!, 0.0, accuracy: 0.3)
    }

    // A single SeededRandomNumberGenerator threaded through two different methods
    // produces independent reproducible draws
    func testCrossMethodSeededComposition() {
        var a = SeededRandomNumberGenerator(seed: 555)
        let normalA = [Double].randomNormal(50, using: &a)
        let exponentialA = [Double].randomExponential(50, using: &a)

        var b = SeededRandomNumberGenerator(seed: 555)
        let normalB = [Double].randomNormal(50, using: &b)
        let exponentialB = [Double].randomExponential(50, using: &b)

        XCTAssertEqual(normalA, normalB)
        XCTAssertEqual(exponentialA, exponentialB)
    }

    // Composes with Swift stdlib's `Array.shuffled(using:)`
    func testComposesWithStdlibShuffled() {
        var a = SeededRandomNumberGenerator(seed: 17)
        let data = [Double].random(20, using: &a)
        let shuffled = data.shuffled(using: &a)
        XCTAssertEqual(shuffled.sorted(), data.sorted())

        // And reproducibility holds across the whole pipeline
        var b = SeededRandomNumberGenerator(seed: 17)
        let dataB = [Double].random(20, using: &b)
        let shuffledB = dataB.shuffled(using: &b)
        XCTAssertEqual(shuffled, shuffledB)
    }

    // MARK: - Random Integer Tests

    // 1D, 2D, and negative-range int generation all produce values in bounds
    func testRandomInt() {
        let result1D = [Int].random(100, in: 0..<10)
        XCTAssertEqual(result1D.count, 100)
        for value in result1D {
            XCTAssertGreaterThanOrEqual(value, 0)
            XCTAssertLessThan(value, 10)
        }

        let result2D = [Int].random(3, 4, in: 1..<7)
        XCTAssertEqual(result2D.count, 3)
        XCTAssertEqual(result2D[0].count, 4)
        for row in result2D {
            for value in row {
                XCTAssertGreaterThanOrEqual(value, 1)
                XCTAssertLessThan(value, 7)
            }
        }

        let negative = [Int].random(50, in: -10..<0)
        for value in negative {
            XCTAssertGreaterThanOrEqual(value, -10)
            XCTAssertLessThan(value, 0)
        }
    }
}
