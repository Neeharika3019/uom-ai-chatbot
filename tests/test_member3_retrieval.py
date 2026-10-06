import os
import sys


# =========================================================
# Add project root to Python path
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# =========================================================
# Import retriever
# =========================================================

from backend.retriever import retrieve


# =========================================================
# Member 3 test questions
# =========================================================

TEST_CASES = [

    {
        "id": "M301",
        "question": "How can I contact the University of Mauritius?",
        "expected": "M3_CONTACT_001"
    },

    {
        "id": "M302",
        "question": (
            "What is the telephone number for the "
            "Admissions and Student Records Office?"
        ),
        "expected": "M3_CONTACT_002"
    },

    {
        "id": "M303",
        "question": "Where is the Examinations Office located?",
        "expected": "M3_CONTACT_003"
    },

    {
        "id": "M304",
        "question": "How can I contact the UoM Library?",
        "expected": "M3_CONTACT_004"
    },

    {
        "id": "M305",
        "question": "Which faculty offers Cyber Security?",
        "expected": "M3_FAC_003"
    },

    {
        "id": "M306",
        "question": (
            "Which department offers BSc (Hons) Cyber Security?"
        ),
        "expected": "M3_FAC_003"
    },

    {
        "id": "M307",
        "question": (
            "What departments are in the Faculty of Science?"
        ),
        "expected": "M3_FAC_006"
    },

    {
        "id": "M308",
        "question": (
            "What should I do if I miss an examination "
            "because I am sick?"
        ),
        "expected": "M3_EXAM_002"
    },

    {
        "id": "M309",
        "question": "What counts as plagiarism at UoM?",
        "expected": "M3_INT_001"
    },

    {
        "id": "M310",
        "question": (
            "Can I use Artificial Intelligence in my "
            "university work?"
        ),
        "expected": "M3_INT_002"
    },

    {
        "id": "M311",
        "question": (
            "What happens if a student uses an "
            "unauthorised device during an exam?"
        ),
        "expected": "M3_EXAM_001"
    },

    {
        "id": "M312",
        "question": (
            "What is the difference between a resit "
            "and a retake?"
        ),
        "expected": "M3_ASSESS_003"
    },

    {
        "id": "M313",
        "question": (
            "What are the rules for final year projects "
            "and dissertations?"
        ),
        "expected": "M3_FYP_001"
    },

    {
        "id": "M314",
        "question": (
            "Does UoM provide financial assistance "
            "to students?"
        ),
        "expected": "M3_SUPPORT_004"
    },

    {
        "id": "M315",
        "question": (
            "What happens if a student breaks "
            "University rules or discipline?"
        ),
        "expected": "M3_CONDUCT_002"
    },
]


# =========================================================
# Run tests
# =========================================================

def main():

    print(
        "\n"
        + "=" * 72
    )

    print(
        "MEMBER 3 RETRIEVAL TESTS"
    )

    print(
        "=" * 72
    )


    top1_passed = 0
    top3_passed = 0


    for test in TEST_CASES:

        results = retrieve(
            test["question"],
            top_k=3
        )


        retrieved_ids = [
            result["id"]
            for result in results
        ]


        top1_pass = (
            len(retrieved_ids) > 0
            and retrieved_ids[0]
            == test["expected"]
        )


        top3_pass = (
            test["expected"]
            in retrieved_ids
        )


        if top1_pass:
            top1_passed += 1

        if top3_pass:
            top3_passed += 1


        print(
            "\n"
            + "=" * 72
        )

        print(
            f"TEST ID: {test['id']}"
        )

        print(
            f"QUESTION: {test['question']}"
        )

        print(
            f"EXPECTED: {test['expected']}"
        )


        if not results:

            print(
                "RETRIEVED RESULTS: NONE"
            )

        else:

            print(
                "\nRETRIEVED RESULTS:"
            )


            for index, result in enumerate(
                results,
                start=1
            ):

                print(
                    f"{index}. "
                    f"{result['id']} - "
                    f"{result['title']} "
                    f"(score={result['score']:.4f})"
                )


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


    print(
        "\n"
        + "=" * 72
    )

    print(
        "MEMBER 3 RETRIEVAL SUMMARY"
    )

    print(
        f"Top-1 Accuracy: "
        f"{top1_passed}/{len(TEST_CASES)} "
        f"({top1_passed / len(TEST_CASES) * 100:.1f}%)"
    )

    print(
        f"Top-3 Accuracy: "
        f"{top3_passed}/{len(TEST_CASES)} "
        f"({top3_passed / len(TEST_CASES) * 100:.1f}%)"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()