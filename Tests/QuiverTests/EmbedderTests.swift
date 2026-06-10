import XCTest
@testable import Quiver

final class EmbedderTests: XCTestCase {

    // A stub embedder backed by a fixed lookup table. Returns nil for any
    // text not in the table, exercising the drop-on-nil path without a model.
    private struct StubEmbedder: Embedder {
        let table: [String: [Double]]
        func embed(_ text: String) -> [Double]? { table[text] }
    }

    private var stub: StubEmbedder {
        StubEmbedder(table: [
            "a": [1.0, 0.0],
            "b": [0.0, 1.0],
            "c": [1.0, 1.0],
        ])
    }

    // MARK: embed(_:)

    func testEmbedReturnsVectorForKnownText() {
        XCTAssertEqual(stub.embed("a"), [1.0, 0.0])
    }

    func testEmbedReturnsNilForUnknownText() {
        XCTAssertNil(stub.embed("missing"))
    }

    // MARK: embedded(using:)

    func testEmbeddedMapsEveryString() {
        let vectors = ["a", "b", "c"].embedded(using: stub)
        XCTAssertEqual(vectors, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    }

    func testEmbeddedPreservesOrder() {
        let vectors = ["c", "a", "b"].embedded(using: stub)
        XCTAssertEqual(vectors, [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    }

    func testEmbeddedDropsUnembeddableStrings() {
        // "x" and "y" are absent → dropped; result is shorter than input.
        let vectors = ["a", "x", "b", "y"].embedded(using: stub)
        XCTAssertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
    }

    func testEmbeddedOnEmptyArrayReturnsEmpty() {
        let vectors = [String]().embedded(using: stub)
        XCTAssertTrue(vectors.isEmpty)
    }

    func testEmbeddedAllUnembeddableReturnsEmpty() {
        let vectors = ["x", "y", "z"].embedded(using: stub)
        XCTAssertTrue(vectors.isEmpty)
    }

    // MARK: Integration with the similarity surface

    func testEmbeddedFlowsIntoSimilaritySearch() throws {
        let docs = ["a", "b", "c"]
        let vectors = docs.embedded(using: stub)

        // Query identical to "a": cosine 1.0 with a, 0.0 with b, ~0.707 with c.
        let query = try XCTUnwrap(stub.embed("a"))
        let hits = vectors.cosineSimilarities(to: query).topIndices(k: 3, labels: docs)

        XCTAssertEqual(hits.map(\.label), ["a", "c", "b"])
        XCTAssertEqual(hits[0].score, 1.0, accuracy: 1e-10)
        XCTAssertEqual(hits[1].score, 1.0 / 2.0.squareRoot(), accuracy: 1e-10)
        XCTAssertEqual(hits[2].score, 0.0, accuracy: 1e-10)
    }
}
