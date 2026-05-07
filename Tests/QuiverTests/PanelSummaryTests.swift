import XCTest
@testable import Quiver

final class PanelSummaryTests: XCTestCase {

    func testMultiColumnStatsCorrect() throws {
        let panel = Panel([
            ("scores", [68.0, 72.0, 75.0, 77.0, 80.0]),
            ("ages", [25.0, 28.0, 31.0, 34.0, 37.0])
        ])

        let summary = panel.summary()

        let scores = try XCTUnwrap(summary.columns["scores"])
        XCTAssertEqual(scores.count, 5)
        XCTAssertEqual(scores.mean, 74.4, accuracy: 1e-10)
        XCTAssertEqual(scores.min, 68.0)
        XCTAssertEqual(scores.max, 80.0)

        let ages = try XCTUnwrap(summary.columns["ages"])
        XCTAssertEqual(ages.count, 5)
        XCTAssertEqual(ages.mean, 31.0, accuracy: 1e-10)
        XCTAssertEqual(ages.min, 25.0)
        XCTAssertEqual(ages.max, 37.0)
    }

    func testColumnNamesPreservesDeclarationOrder() {
        let panel = Panel([
            ("zeta", [1.0, 2.0]),
            ("alpha", [3.0, 4.0]),
            ("mu", [5.0, 6.0])
        ])

        XCTAssertEqual(panel.summary().columnNames, ["zeta", "alpha", "mu"])
    }

    func testDescriptionContainsExpectedTableElements() {
        let panel = Panel([
            ("x", [1.0, 2.0, 3.0, 4.0, 5.0])
        ])
        let text = panel.summary().description

        // Headers
        XCTAssertTrue(text.contains("column"))
        XCTAssertTrue(text.contains("count"))
        XCTAssertTrue(text.contains("mean"))
        XCTAssertTrue(text.contains("std"))
        XCTAssertTrue(text.contains("min"))
        XCTAssertTrue(text.contains("max"))
        // Data
        XCTAssertTrue(text.contains("x"))
        XCTAssertTrue(text.contains("3.0"))
        XCTAssertTrue(text.contains("1.0"))
        XCTAssertTrue(text.contains("5.0"))
        // Separator row exists
        XCTAssertTrue(text.contains("---"))
    }

    func testMarkdownTableHasColumnHeadersAndStatRows() {
        let panel = Panel([
            ("scores", [68.0, 72.0, 75.0]),
            ("ages", [25.0, 28.0, 31.0])
        ])
        let table = panel.summary().markdownTable()

        XCTAssertTrue(table.contains("| Statistic |"))
        XCTAssertTrue(table.contains("| scores |"))
        XCTAssertTrue(table.contains("| ages |"))
        XCTAssertTrue(table.contains("| count |"))
        XCTAssertTrue(table.contains("| mean |"))
    }

    func testCsvRowsHasOneRowPerColumn() {
        let panel = Panel([
            ("a", [1.0, 2.0, 3.0]),
            ("b", [4.0, 5.0, 6.0])
        ])
        let csv = panel.summary().csvRows()

        let lines = csv.split(separator: "\n")
        // Header + 2 columns = 3 lines
        XCTAssertEqual(lines.count, 3)
        XCTAssertEqual(lines.first, "column,count,mean,std,min,max")
        XCTAssertTrue(csv.contains("a,3,"))
        XCTAssertTrue(csv.contains("b,3,"))
    }

    func testCodableRoundTrip() throws {
        let panel = Panel([
            ("scores", [68.0, 72.0, 75.0, 77.0, 80.0])
        ])
        let original = panel.summary()

        let encoded = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(PanelSummary.self, from: encoded)

        XCTAssertEqual(original, restored)
    }

    func testEquatable() {
        let p1 = Panel([("scores", [1.0, 2.0, 3.0])])
        let p2 = Panel([("scores", [1.0, 2.0, 3.0])])
        let p3 = Panel([("scores", [1.0, 2.0, 99.0])])

        XCTAssertEqual(p1.summary(), p2.summary())
        XCTAssertNotEqual(p1.summary(), p3.summary())
    }
}
