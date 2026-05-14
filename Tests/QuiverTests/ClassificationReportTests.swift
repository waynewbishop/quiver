import XCTest
@testable import Quiver

final class ClassificationReportTests: XCTestCase {

    func testBalancedBinaryStructuredAccess() throws {
        // 8 predictions vs ground truth, accuracy = 6/8 = 0.75
        let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
        let actual      = [1, 0, 0, 1, 0, 1, 1, 0]

        let report = predictions.classificationReport(actual: actual)

        XCTAssertEqual(report.classOrder, [0, 1])
        XCTAssertEqual(report.totalSupport, 8)
        XCTAssertEqual(report.accuracy, 0.75, accuracy: 1e-10)

        let class0 = try XCTUnwrap(report.perClass[0])
        XCTAssertEqual(class0.support, 4)
        XCTAssertEqual(try XCTUnwrap(class0.precision), 0.75, accuracy: 1e-10)
        XCTAssertEqual(try XCTUnwrap(class0.recall), 0.75, accuracy: 1e-10)

        let class1 = try XCTUnwrap(report.perClass[1])
        XCTAssertEqual(class1.support, 4)
        XCTAssertEqual(try XCTUnwrap(class1.precision), 0.75, accuracy: 1e-10)
        XCTAssertEqual(try XCTUnwrap(class1.recall), 0.75, accuracy: 1e-10)
    }

    func testNoPredictedPositivesProducesNilPrecision() {
        // Predicting all 0s when half the actual are 1s.
        // For class 1: TP=0, FP=0 — precision is undefined.
        let predictions = [0, 0, 0, 0]
        let actual      = [1, 1, 0, 0]

        let report = predictions.classificationReport(actual: actual)

        let class1 = report.perClass[1]
        XCTAssertNotNil(class1)
        XCTAssertNil(class1?.precision, "Precision must be nil when no positives predicted")
    }

    func testMacroAndWeightedAverages() throws {
        // Multi-class case with all metrics defined
        let predictions = [0, 1, 2, 0, 1, 2]
        let actual      = [0, 1, 2, 0, 2, 1]

        let report = predictions.classificationReport(actual: actual)

        // Macro averages of three classes with precision values 1.0, 0.5, 0.5
        let macroP = try XCTUnwrap(report.macroAverage.precision)
        XCTAssertEqual(macroP, 2.0 / 3.0, accuracy: 1e-10)

        // Weighted averages with equal supports (2 each) reduce to the macro average
        let weightedP = try XCTUnwrap(report.weightedAverage.precision)
        XCTAssertEqual(weightedP, 2.0 / 3.0, accuracy: 1e-10)
    }

    func testDescriptionContainsExpectedElements() {
        let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
        let actual      = [1, 0, 0, 1, 0, 1, 1, 0]
        let text = predictions.classificationReport(actual: actual).description

        XCTAssertTrue(text.contains("precision"))
        XCTAssertTrue(text.contains("recall"))
        XCTAssertTrue(text.contains("f1-score"))
        XCTAssertTrue(text.contains("support"))
        XCTAssertTrue(text.contains("accuracy"))
        XCTAssertTrue(text.contains("macro avg"))
        XCTAssertTrue(text.contains("weighted avg"))
        XCTAssertTrue(text.contains("0.75"))
    }

    func testMarkdownTableStructure() {
        let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
        let actual      = [1, 0, 0, 1, 0, 1, 1, 0]
        let table = predictions.classificationReport(actual: actual).markdownTable()

        XCTAssertTrue(table.contains("| Class |"))
        XCTAssertTrue(table.contains("| Precision |"))
        XCTAssertTrue(table.contains("| Recall |"))
        XCTAssertTrue(table.contains("| F1 |"))
        XCTAssertTrue(table.contains("| Support |"))
        XCTAssertTrue(table.contains("| accuracy |"))
        XCTAssertTrue(table.contains("| macro avg |"))
        XCTAssertTrue(table.contains("| weighted avg |"))
    }

    func testCsvRowsHasOneRowPerClassPlusAggregates() {
        let predictions = [1, 0, 1, 1, 0, 0, 1, 0]
        let actual      = [1, 0, 0, 1, 0, 1, 1, 0]
        let csv = predictions.classificationReport(actual: actual).csvRows()

        let lines = csv.split(separator: "\n")
        // Header + 2 classes + accuracy + macro + weighted = 6 lines
        XCTAssertEqual(lines.count, 6)
        XCTAssertEqual(lines.first, "class,precision,recall,f1,support")
        XCTAssertTrue(csv.contains("accuracy"))
        XCTAssertTrue(csv.contains("macro avg"))
        XCTAssertTrue(csv.contains("weighted avg"))
    }

    func testCodableRoundTripPreservesOptionalFields() throws {
        // Includes a class with nil precision (undefined)
        let predictions = [0, 0, 0, 0]
        let actual      = [1, 1, 0, 0]

        let original = predictions.classificationReport(actual: actual)

        let encoded = try JSONEncoder().encode(original)
        let restored = try JSONDecoder().decode(ClassificationReport.self, from: encoded)

        XCTAssertEqual(original, restored)
        XCTAssertNil(restored.perClass[1]?.precision, "Optional precision should round-trip as nil")
    }

    func testEquatable() {
        let a = [1, 0, 1, 0].classificationReport(actual: [1, 0, 1, 0])
        let b = [1, 0, 1, 0].classificationReport(actual: [1, 0, 1, 0])
        let c = [1, 0, 1, 0].classificationReport(actual: [0, 1, 0, 1])

        XCTAssertEqual(a, b)
        XCTAssertNotEqual(a, c)
    }
}
