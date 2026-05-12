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

import Foundation

// MARK: - Seeded Random Number Generator

/// A deterministic random number generator that produces the same sequence for a given seed.
///
/// Swift's built-in `SystemRandomNumberGenerator` is intentionally non-reproducible.
/// Apple provides the `RandomNumberGenerator` protocol so callers can plug in their own
/// generator when reproducibility is required — for tutorials with stable example output,
/// unit tests of stochastic code, or any "same seed, same answer" workflow.
///
/// The generator uses the xorshift64 algorithm: three XOR-shift operations on a `UInt64`
/// state. Same seed always produces the same sequence of numbers. A seed of `0` is
/// remapped to `1` internally, since xorshift cannot escape an all-zero state.
///
/// Use it anywhere Swift's standard library accepts a `RandomNumberGenerator`. Quiver's
/// random methods expose `using:` overloads that mirror the standard library's
/// `Array.shuffled(using:)` pattern, so the same generator threads through Quiver and
/// stdlib calls without ceremony:
///
/// ```swift
/// var rng = SeededRandomNumberGenerator(seed: 42)
/// let data = [Double].randomNormal(1_000, mean: 100, standardDeviation: 15, using: &rng)
/// let shuffled = data.shuffled(using: &rng)
/// ```
///
/// - Note: The xorshift64 algorithm is suitable for simulations, teaching, and reproducible
///   examples. It is **not** a cryptographic generator. For security-sensitive randomness,
///   use Swift's `SystemRandomNumberGenerator`.
public struct SeededRandomNumberGenerator: RandomNumberGenerator {
    private var state: UInt64

    /// The seed originally supplied to the generator.
    ///
    /// The stored value reflects the caller's input — it is not updated as the generator
    /// advances. Two generators created with the same `seed` produce identical sequences.
    public let seed: UInt64

    /// Creates a generator seeded with the given value.
    ///
    /// - Parameter seed: A `UInt64` seed. A seed of `0` is internally remapped to `1` so
    ///   the generator does not produce an all-zero sequence.
    public init(seed: UInt64) {
        self.seed = seed
        // A zero state would produce all zeros, so we default to 1
        self.state = seed == 0 ? 1 : seed
    }

    /// Returns the next random `UInt64` by scrambling the internal state.
    public mutating func next() -> UInt64 {
        state ^= state << 13
        state ^= state >> 7
        state ^= state << 17
        return state
    }
}
