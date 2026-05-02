import Foundation

// MARK: - Embedding Dictionary Nearest-Neighbor Search

public extension Dictionary where Key == String, Value == [Double] {

    /// Returns the top K words in the embedding dictionary nearest to a query vector.
    ///
    /// Computes cosine similarity between the query vector and every value in the dictionary,
    /// then returns the highest-scoring entries in descending order. Commonly used with word
    /// embedding tables (GloVe, Word2Vec, FastText) to find semantically similar words, run
    /// analogy queries, and rank candidates by directional alignment.
    ///
    /// Example:
    /// ```swift
    /// let king  = [0.9, 0.2, 0.8, 0.7]
    /// let queen = [0.3, 0.9, 0.8, 0.7]
    /// let man   = [0.8, 0.1, 0.2, 0.6]
    /// let woman = [0.2, 0.8, 0.2, 0.6]
    ///
    /// let embeddings: [String: [Double]] = [
    ///     "king":  king,
    ///     "queen": queen,
    ///     "man":   man,
    ///     "woman": woman
    /// ]
    ///
    /// // king - man + woman should land closest to queen
    /// let target = king.subtract(man).add(woman)
    /// let results = embeddings.nearest(to: target, k: 2)
    /// // [(rank: 1, word: "queen", score: 1.0),
    /// //  (rank: 2, word: "king", score: 0.79)]
    /// ```
    ///
    /// Entries whose vector dimensions do not match the query are skipped. Entries with
    /// zero-magnitude vectors score 0.0 (perpendicular by convention) rather than producing
    /// NaN, matching the behavior of `cosineOfAngle(with:)`.
    ///
    /// - Complexity: O(*n* log *n*) where *n* is the dictionary size, dominated by the sort.
    ///   Each similarity is O(*d*) where *d* is the vector dimension.
    /// - Parameters:
    ///   - query: The reference vector to compare every dictionary entry against.
    ///   - k: Number of top results to return. Default is 5.
    /// - Returns: Array of tuples containing rank (1-based), word, and similarity score,
    ///   sorted by score (highest first). Returns fewer than `k` entries if the dictionary
    ///   has fewer matching-dimension values.
    func nearest(to query: [Double], k: Int = 5) -> [(rank: Int, word: String, score: Double)] {
        precondition(k > 0, "k must be positive")
        precondition(!query.isEmpty, "Query vector must not be empty")

        // Score every entry whose vector dimension matches the query. Skip mismatches
        // silently — embedding dictionaries occasionally include entries from a
        // different model with a different dimensionality.
        var scored: [(word: String, score: Double)] = []
        scored.reserveCapacity(self.count)
        for (word, vector) in self {
            guard vector.count == query.count else { continue }
            scored.append((word: word, score: vector.cosineOfAngle(with: query)))
        }

        return scored
            .sorted { $0.score > $1.score }
            .prefix(k)
            .enumerated()
            .map { (rank: $0.offset + 1, word: $0.element.word, score: $0.element.score) }
    }
}
