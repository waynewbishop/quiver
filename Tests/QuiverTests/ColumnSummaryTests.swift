import XCTest
@testable import Quiver

final class ColumnSummaryTests: XCTestCase {

    func testAllFieldsPopulated() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        let s = try XCTUnwrap(data.summary())

        XCTAssertEqual(s.count, 10)
        XCTAssertEqual(s.mean, 5.5, accuracy: 1e-10)
        XCTAssertEqual(s.min, 1.0)
        XCTAssertEqual(s.max, 10.0)
        XCTAssertEqual(s.median, 5.5, accuracy: 1e-10)
        XCTAssertEqual(s.q1, 3.25, accuracy: 1e-10)
        XCTAssertEqual(s.q3, 7.75, accuracy: 1e-10)
        XCTAssertEqual(s.iqr, 4.5, accuracy: 1e-10)
        // Sample std for 1..10
        XCTAssertEqual(s.std, 3.0276503540974917, accuracy: 1e-10)
    }

    func testReturnsNilOnEmptyArray() {
        let empty: [Double] = []
        XCTAssertNil(empty.summary())
    }

    func testDescriptionContainsAllNineLabels() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0]
        let s = try XCTUnwrap(data.summary())
        let text = s.description

        for label in ["count:", "mean:", "std:", "min:", "q1:", "median:", "q3:", "max:", "iqr:"] {
            XCTAssertTrue(text.contains(label), "description missing label \(label)")
        }
    }

    func testMarkdownTableStructure() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0]
        let s = try XCTUnwrap(data.summary())
        let table = s.markdownTable()

        // Header + separator + 9 stat rows = 11 lines
        let lines = table.split(separator: "\n")
        XCTAssertEqual(lines.count, 11)
        XCTAssertEqual(lines.first, "| Statistic | Value |")
        XCTAssertEqual(lines[1], "| --- | --- |")
        XCTAssertTrue(table.contains("count"))
        XCTAssertTrue(table.contains("iqr"))
    }

    func testCsvRowsStructure() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0]
        let s = try XCTUnwrap(data.summary())
        let csv = s.csvRows()

        // Header + 9 stat rows = 10 lines
        let lines = csv.split(separator: "\n")
        XCTAssertEqual(lines.count, 10)
        XCTAssertEqual(lines.first, "statistic,value")
        XCTAssertTrue(csv.contains("count,5"))
    }

    func testCodableRoundTrip() throws {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0]
        let original = try XCTUnwrap(data.summary())

        let encoded = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(ColumnSummary.self, from: encoded)

        XCTAssertEqual(original, restored)
    }

    func testEquatable() throws {
        let a = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 5.0].summary())
        let b = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 5.0].summary())
        let c = try XCTUnwrap([1.0, 2.0, 3.0, 4.0, 100.0].summary())

        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}
