// Copyright 2025 Wayne W Bishop. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation

public extension String {

    /// Tokenizes text into individual words for text analysis and embedding workflows.
    ///
    /// This method converts text to lowercase, splits on whitespace and newlines,
    /// and filters out empty strings. By default, punctuation is stripped from each
    /// token so that words like "running!" match their embedding key "running".
    ///
    /// Punctuation stripping removes leading and trailing characters from the
    /// Unicode general categories for punctuation (e.g., periods, commas, quotes,
    /// exclamation marks, question marks, hyphens, brackets, and other symbols
    /// classified as punctuation). Interior punctuation in contractions like
    /// "don't" is preserved because the apostrophe is not at the boundary after
    /// the first trim pass.
    ///
    /// Example:
    /// ```swift
    /// // Default: punctuation stripped for clean embedding lookups
    /// let text = "Hello, world! How are you?"
    /// let words = text.tokenize()  // ["hello", "world", "how", "are", "you"]
    ///
    /// // Preserve punctuation when token boundaries matter
    /// let raw = text.tokenize(strippingPunctuation: false)
    /// // ["hello,", "world!", "how", "are", "you?"]
    /// ```
    ///
    /// - Parameter strippingPunctuation: When `true` (the default), leading and
    ///   trailing punctuation characters are removed from each token. Set to `false`
    ///   to preserve punctuation as part of the token.
    /// - Returns: An array of lowercase word tokens with empty strings removed
    func tokenize(strippingPunctuation: Bool = true) -> [String] {
        let tokens = self.lowercased()
            .components(separatedBy: .whitespacesAndNewlines)
            .filter { !$0.isEmpty }

        guard strippingPunctuation else {
            return tokens
        }

        return tokens
            .map { $0.trimmingCharacters(in: .punctuationCharacters) }
            .filter { !$0.isEmpty }
    }
}

public extension Array where Element == String {

    /// Converts an array of words to their vector embeddings by looking up each word in the embeddings dictionary.
    ///
    /// This method looks up each word in the provided embeddings dictionary and returns only
    /// the vectors for words that exist in the dictionary. Words not found are automatically
    /// filtered out, making it safe to use with any vocabulary.
    ///
    /// Example:
    /// ```swift
    /// let words = ["running", "shoes", "unknown"]
    /// let embeddings = [
    ///     "running": [0.8, 0.7, 0.9],
    ///     "shoes": [0.1, 0.9, 0.2]
    /// ]
    /// let vectors = words.embed(using: embeddings)
    /// // Returns: [[0.8, 0.7, 0.9], [0.1, 0.9, 0.2]]
    /// // "unknown" is filtered out
    /// ```
    ///
    /// - Parameter embeddings: Dictionary mapping words to their vector representations
    /// - Returns: Array of vectors for words found in the embeddings dictionary
    func embed(using embeddings: [String: [Double]]) -> [[Double]] {
        return self.compactMap { embeddings[$0] }
    }
}
