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
# Member 2 retrieval test cases
# =========================================================

TEST_CASES = [

    {
        "id": "M201",
        "question": (
            "When do Semester 1 lectures start at UoM?"
        ),
        "expected": "ACAD_001"
    },

    {
        "id": "M202",
        "question": (
            "What is the deadline for module registration "
            "in Semester 1?"
        ),
        "expected": "ACAD_002"
    },

    {
        "id": "M203",
        "question": (
            "When does Floating Week take place "
            "in Semester 1?"
        ),
        "expected": "ACAD_003"
    },

    {
        "id": "M204",
        "question": (
            "When do Semester 1 examinations "
            "start and end?"
        ),
        "expected": "ACAD_005"
    },

    {
        "id": "M205",
        "question": (
            "When do Semester 2 lectures start "
            "for all students?"
        ),
        "expected": "ACAD_007"
    },

    {
        "id": "M206",
        "question": (
            "When is Students' Week held "
            "in Semester 2?"
        ),
        "expected": "ACAD_009"
    },

    {
        "id": "M207",
        "question": (
            "When do Semester 2 examinations "
            "start and end?"
        ),
        "expected": "ACAD_011"
    },

    {
        "id": "M208",
        "question": (
            "What is the penalty fee deadline "
            "for late module registration "
            "in Semester 1?"
        ),
        "expected": "ACAD_004"
    },

    {
        "id": "M209",
        "question": (
            "When are Graduation Ceremonies "
            "scheduled in 2027?"
        ),
        "expected": "ACAD_012"
    },

    {
        "id": "M210",
        "question": (
            "What are the UoM library opening "
            "hours during term time?"
        ),
        "expected": "SERV_001"
    },

    {
        "id": "M211",
        "question": (
            "What are the library opening hours "
            "during the vacation period?"
        ),
        "expected": "SERV_002"
    },

    {
        "id": "M212",
        "question": (
            "How many books can undergraduate "
            "students borrow from the library?"
        ),
        "expected": "SERV_003"
    },

    {
        "id": "M213",
        "question": (
            "How many books can Academic Staff "
            "borrow from the library?"
        ),
        "expected": "SERV_004"
    },

    {
        "id": "M214",
        "question": (
            "Where is the main CITS Computer "
            "Lab located?"
        ),
        "expected": "SERV_008"
    },

    {
        "id": "M215",
        "question": (
            "Where are CITS Labs 1A, 1B, "
            "and 1C located?"
        ),
        "expected": "SERV_009"
    },

    {
        "id": "M216",
        "question": (
            "What is the MCB bank account "
            "number for UoM payments?"
        ),
        "expected": "SERV_012"
    },

    {
        "id": "M217",
        "question": (
            "What digital and mobile payment "
            "apps are accepted by UoM?"
        ),
        "expected": "SERV_015"
    },

    {
        "id": "M218",
        "question": (
            "What is the tuition fee for studying "
            "Computer Science at Harvard University?"
        ),
        "expected": None
    },

    {
        "id": "M219",
        "question": (
            "Who is the Prime Minister of Australia?"
        ),
        "expected": None
    },

    {
        "id": "M220",
        "question": (
            "Does UoM offer free accommodation "
            "and flights to international students?"
        ),
        "expected": None
    }
]


# =========================================================
# Run evaluation
# =========================================================

def main():

    passed = 0

    total = len(
        TEST_CASES
    )


    print(
        "\n"
        + "=" * 75
    )

    print(
        "MEMBER 2 RETRIEVAL INTEGRATION TEST"
    )

    print(
        "=" * 75
    )


    for test in TEST_CASES:

        question = test[
            "question"
        ]

        expected = test[
            "expected"
        ]


        results = retrieve(
            question,
            top_k=3
        )


        print(
            "\n"
            + "-" * 75
        )

        print(
            f"TEST: {test['id']}"
        )

        print(
            f"QUESTION: {question}"
        )

        print(
            f"EXPECTED: "
            f"{expected if expected else 'NO RESULT'}"
        )


        # =================================================
        # Off-topic / unsupported question
        # =================================================

        if expected is None:

            if not results:

                print(
                    "RESULT: NO RECORDS"
                )

                print(
                    "PASS"
                )

                passed += 1

            else:

                print(
                    "RETRIEVED:"
                )

                for result in results:

                    print(
                        f"  {result['id']} - "
                        f"{result['title']} "
                        f"(score={result['score']:.4f})"
                    )

                print(
                    "FAIL"
                )


            continue


        # =================================================
        # Expected record
        # =================================================

        if not results:

            print(
                "RESULT: NO RECORDS"
            )

            print(
                "FAIL"
            )

            continue


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


        top_result = results[0][
            "id"
        ]


        if top_result == expected:

            print(
                "TOP-1: PASS"
            )

            passed += 1

        else:

            print(
                "TOP-1: FAIL"
            )


    # =====================================================
    # Summary
    # =====================================================

    accuracy = (
        passed
        / total
        * 100
    )


    print(
        "\n"
        + "=" * 75
    )

    print(
        "MEMBER 2 RETRIEVAL TEST SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"Passed: {passed}/{total}"
    )

    print(
        f"Accuracy: {accuracy:.1f}%"
    )

    print(
        "=" * 75
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()