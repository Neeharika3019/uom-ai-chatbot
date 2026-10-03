import os
import sys
import pandas as pd


# =========================================================
# Add project root to Python path
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(
    PROJECT_ROOT
)


from backend.retriever import retrieve


# =========================================================
# Configuration
# =========================================================

TEST_FILE = os.path.join(
    PROJECT_ROOT,
    "tests",
    "retrieval_test_questions.xlsx"
)

TOP_K = 3


# =========================================================
# Load test dataset
# =========================================================

print(
    "Loading retrieval test dataset..."
)

test_df = pd.read_excel(
    TEST_FILE
)

test_df.columns = (
    test_df.columns
    .str.strip()
)

test_df = test_df.fillna("")


print(
    f"Total test questions loaded: "
    f"{len(test_df)}"
)


# =========================================================
# Metrics
# =========================================================

single_total = 0
single_top1_correct = 0
single_top3_correct = 0


multi_total = 0
multi_full_coverage = 0
multi_coverage_scores = []


off_topic_total = 0
off_topic_rejected = 0


# =========================================================
# Run evaluation
# =========================================================

for _, row in test_df.iterrows():

    test_id = str(
        row.get(
            "Test ID",
            ""
        )
    ).strip()

    question = str(
        row.get(
            "Question",
            ""
        )
    ).strip()

    expected_value = str(
        row.get(
            "Expected Record ID",
            ""
        )
    ).strip()

    test_type = str(
        row.get(
            "Test Type",
            ""
        )
    ).strip()


    # -----------------------------------------------------
    # Convert expected IDs into list
    # -----------------------------------------------------

    if expected_value.upper() in [
        "",
        "N/A",
        "NA",
        "NONE"
    ]:

        expected_ids = []

    else:

        expected_ids = [
            value.strip()
            for value
            in expected_value.split("/")
            if value.strip()
        ]


    # -----------------------------------------------------
    # Retrieve
    # -----------------------------------------------------

    results = retrieve(
        question,
        top_k=TOP_K
    )

    retrieved_ids = [
        result["id"]
        for result in results
    ]


    # =====================================================
    # Display
    # =====================================================

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"TEST ID: {test_id}"
    )

    print(
        f"QUESTION: {question}"
    )

    print(
        "EXPECTED RECORD(S): "
        f"{expected_ids if expected_ids else 'N/A'}"
    )

    print(
        f"TEST TYPE: {test_type}"
    )


    if results:

        print(
            "RETRIEVAL MODE: "
            f"{results[0]['retrieval_mode']}"
        )

    else:

        print(
            "RETRIEVAL MODE: NONE"
        )


    print(
        "\nRETRIEVED RESULTS:"
    )


    if not results:

        print(
            "No records retrieved."
        )

    else:

        for number, result in enumerate(
            results,
            start=1
        ):

            print(
                f"{number}. "
                f"{result['id']} - "
                f"{result['title']} "
                f"(score={result['score']:.4f})"
            )


    # =====================================================
    # OFF-TOPIC TEST
    # =====================================================

    if len(expected_ids) == 0:

        off_topic_total += 1

        if len(results) == 0:

            off_topic_rejected += 1

            print(
                "\nOFF-TOPIC: PASS"
            )

        else:

            print(
                "\nOFF-TOPIC: FAIL"
            )

        continue


    # =====================================================
    # SINGLE-ANSWER TEST
    # =====================================================

    if len(expected_ids) == 1:

        single_total += 1

        expected_id = expected_ids[0]


        # -------------------------------------------------
        # Top-1
        # -------------------------------------------------

        top1_pass = (
            len(retrieved_ids) > 0
            and retrieved_ids[0]
            == expected_id
        )


        if top1_pass:

            single_top1_correct += 1


        # -------------------------------------------------
        # Top-3
        # -------------------------------------------------

        top3_ids = (
            retrieved_ids[:3]
        )


        top3_pass = (
            expected_id
            in top3_ids
        )


        if top3_pass:

            single_top3_correct += 1


        print(
            "\nTop-1:",
            "PASS"
            if top1_pass
            else "FAIL"
        )

        print(
            "Top-3:",
            "PASS"
            if top3_pass
            else "FAIL"
        )

        continue


    # =====================================================
    # MULTI-ANSWER TEST
    # =====================================================

    multi_total += 1


    expected_set = set(
        expected_ids
    )

    retrieved_set = set(
        retrieved_ids
    )


    correct_matches = (
        expected_set.intersection(
            retrieved_set
        )
    )


    coverage = (
        len(correct_matches)
        / len(expected_set)
        * 100
    )


    multi_coverage_scores.append(
        coverage
    )


    full_coverage = (
        expected_set
        .issubset(
            retrieved_set
        )
    )


    if full_coverage:

        multi_full_coverage += 1


    print(
        f"\nMATCHED: "
        f"{len(correct_matches)}"
        f"/"
        f"{len(expected_set)}"
    )

    print(
        f"COVERAGE: "
        f"{coverage:.1f}%"
    )

    print(
        "FULL COVERAGE:",
        "PASS"
        if full_coverage
        else "FAIL"
    )


# =========================================================
# FINAL EVALUATION
# =========================================================

print(
    "\n"
    + "=" * 75
)

print(
    "FINAL RETRIEVAL EVALUATION"
)


# =========================================================
# Single-answer metrics
# =========================================================

if single_total > 0:

    top1_accuracy = (
        single_top1_correct
        / single_total
        * 100
    )

    top3_accuracy = (
        single_top3_correct
        / single_total
        * 100
    )


    print(
        "\n--- SINGLE-ANSWER QUESTIONS ---"
    )

    print(
        f"Questions: "
        f"{single_total}"
    )

    print(
        f"Top-1 Accuracy: "
        f"{single_top1_correct}"
        f"/"
        f"{single_total} "
        f"({top1_accuracy:.1f}%)"
    )

    print(
        f"Top-3 Accuracy: "
        f"{single_top3_correct}"
        f"/"
        f"{single_total} "
        f"({top3_accuracy:.1f}%)"
    )


# =========================================================
# Multi-answer metrics
# =========================================================

if multi_total > 0:

    average_coverage = (
        sum(
            multi_coverage_scores
        )
        / len(
            multi_coverage_scores
        )
    )


    full_coverage_accuracy = (
        multi_full_coverage
        / multi_total
        * 100
    )


    print(
        "\n--- MULTI-ANSWER QUESTIONS ---"
    )

    print(
        f"Questions: "
        f"{multi_total}"
    )

    print(
        f"Full Coverage: "
        f"{multi_full_coverage}"
        f"/"
        f"{multi_total} "
        f"({full_coverage_accuracy:.1f}%)"
    )

    print(
        f"Average Coverage: "
        f"{average_coverage:.1f}%"
    )


# =========================================================
# Off-topic metrics
# =========================================================

if off_topic_total > 0:

    rejection_accuracy = (
        off_topic_rejected
        / off_topic_total
        * 100
    )


    print(
        "\n--- OFF-TOPIC QUESTIONS ---"
    )

    print(
        f"Questions: "
        f"{off_topic_total}"
    )

    print(
        f"Rejection Accuracy: "
        f"{off_topic_rejected}"
        f"/"
        f"{off_topic_total} "
        f"({rejection_accuracy:.1f}%)"
    )