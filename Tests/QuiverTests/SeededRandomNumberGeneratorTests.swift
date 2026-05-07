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

final class SeededRandomNumberGeneratorTests: XCTestCase {

    // The seed property round-trips the caller's input
    func testSeedRoundTrip() {
        let rng = SeededRandomNumberGenerator(seed: 42)
        XCTAssertEqual(rng.seed, 42)

        let zeroSeeded = SeededRandomNumberGenerator(seed: 0)
        XCTAssertEqual(zeroSeeded.seed, 0)
    }

    // Same seed produces the same sequence of UInt64 values
    func testSameSeedReproducible() {
        var a = SeededRandomNumberGenerator(seed: 12345)
        var b = SeededRandomNumberGenerator(seed: 12345)
        for _ in 0..<100 {
            XCTAssertEqual(a.next(), b.next())
        }
    }

    // Different seeds produce different sequences
    func testDifferentSeedsDiffer() {
        var a = SeededRandomNumberGenerator(seed: 1)
        var b = SeededRandomNumberGenerator(seed: 2)
        var differingPositions = 0
        for _ in 0..<100 {
            if a.next() != b.next() { differingPositions += 1 }
        }
        // Expect nearly all positions to differ; require at least 90 of 100
        XCTAssertGreaterThan(differingPositions, 90)
    }

    // A seed of 0 still produces a non-zero sequence
    func testZeroSeedProducesNonZeroSequence() {
        var rng = SeededRandomNumberGenerator(seed: 0)
        var sawNonZero = false
        for _ in 0..<10 {
            if rng.next() != 0 { sawNonZero = true; break }
        }
        XCTAssertTrue(sawNonZero)
    }

    // Composes with Swift stdlib's `Array.shuffled(using:)`
    func testComposesWithStdlibShuffled() {
        var a = SeededRandomNumberGenerator(seed: 7)
        var b = SeededRandomNumberGenerator(seed: 7)
        let input = Array(0..<20)
        let shuffledA = input.shuffled(using: &a)
        let shuffledB = input.shuffled(using: &b)
        XCTAssertEqual(shuffledA, shuffledB)
        XCTAssertEqual(shuffledA.sorted(), input)
    }
}
