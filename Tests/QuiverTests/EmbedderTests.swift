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

    func testEmbeddedPairsEveryString() {
        let embedded = ["a", "b", "c"].embedded(using: stub)
        XCTAssertEqual(embedded.map(\.text), ["a", "b", "c"])
        XCTAssertEqual(embedded.map(\.vector), [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    }

    func testEmbeddedPreservesOrder() {
        let embedded = ["c", "a", "b"].embedded(using: stub)
        XCTAssertEqual(embedded.map(\.text), ["c", "a", "b"])
        XCTAssertEqual(embedded.map(\.vector), [[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    }

    func testEmbeddedKeepsTextAlignedWhenStringsDrop() {
        // "x" and "y" are absent → dropped. Each surviving vector stays paired
        // with its own text, so the drops can't misalign "b" onto "a"'s vector.
        let embedded = ["a", "x", "b", "y"].embedded(using: stub)
        XCTAssertEqual(embedded.map(\.text), ["a", "b"])
        XCTAssertEqual(embedded.map(\.vector), [[1.0, 0.0], [0.0, 1.0]])
    }

    func testEmbeddedOnEmptyArrayReturnsEmpty() {
        let embedded = [String]().embedded(using: stub)
        XCTAssertTrue(embedded.isEmpty)
    }

    func testEmbeddedAllUnembeddableReturnsEmpty() {
        let embedded = ["x", "y", "z"].embedded(using: stub)
        XCTAssertTrue(embedded.isEmpty)
    }

    // MARK: Integration with the similarity surface

    func testEmbeddedFlowsIntoSimilaritySearch() throws {
        let docs = ["a", "b", "c"]
        let embedded = docs.embedded(using: stub)

        // Query identical to "a": cosine 1.0 with a, 0.0 with b, ~0.707 with c.
        let query = try XCTUnwrap(stub.embed("a"))
        let hits = embedded.mostSimilar(to: query, k: 3)

        XCTAssertEqual(hits.map(\.text), ["a", "c", "b"])
        XCTAssertEqual(hits[0].score, 1.0, accuracy: 1e-10)
        XCTAssertEqual(hits[1].score, 1.0 / 2.0.squareRoot(), accuracy: 1e-10)
        XCTAssertEqual(hits[2].score, 0.0, accuracy: 1e-10)
    }

    func testMostSimilarKeepsTextAlignedAfterDrops() throws {
        // "x" drops out; "b" must not inherit "a"'s position in the results.
        let embedded = ["a", "x", "b"].embedded(using: stub)

        // Query identical to "b": cosine 1.0 with b, 0.0 with a.
        let query = try XCTUnwrap(stub.embed("b"))
        let hits = embedded.mostSimilar(to: query, k: 2)

        XCTAssertEqual(hits.map(\.text), ["b", "a"])
        XCTAssertEqual(hits[0].score, 1.0, accuracy: 1e-10)
        XCTAssertEqual(hits[1].score, 0.0, accuracy: 1e-10)
    }
}
