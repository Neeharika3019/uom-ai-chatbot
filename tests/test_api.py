import os
import sys

from fastapi.testclient import TestClient


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
# Import FastAPI application
# =========================================================

from backend.api import app


# =========================================================
# Create API test client
# =========================================================

client = TestClient(app)


# =========================================================
# Test root endpoint
# =========================================================
def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["service"] == "UoM AI Chatbot API"
    assert data["version"] == "2.0.0"
    assert data["status"] == "running"


# =========================================================
# Test health endpoint
# =========================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["retriever"] == "available"


# =========================================================
# Test normal relevant retrieval
# =========================================================

def test_relevant_question():

    response = client.post(
        "/retrieve",
        json={
            "question": (
                "What cybersecurity programmes are available?"
            ),
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["has_results"] is True

    assert data["result_count"] >= 2

    result_ids = [
        result["id"]
        for result in data["results"]
    ]

    assert "PROG_006" in result_ids

    assert "PROG_007" in result_ids


# =========================================================
# Test metadata retrieval
# =========================================================

def test_metadata_list_question():

    response = client.post(
        "/retrieve",
        json={
            "question": (
                "Show all undergraduate programmes "
                "in FOICDT."
            ),
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["has_results"] is True

    assert data["retrieval_mode"] == "metadata"

    assert data["result_count"] == 7


# =========================================================
# Test off-topic rejection
# =========================================================

def test_off_topic_question():

    response = client.post(
        "/retrieve",
        json={
            "question": "What is the weather tomorrow?",
            "top_k": 3
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert data["has_results"] is False

    assert data["result_count"] == 0

    assert data["results"] == []


# =========================================================
# Test input validation
# =========================================================

def test_empty_question_validation():

    response = client.post(
        "/retrieve",
        json={
            "question": "",
            "top_k": 3
        }
    )

    # FastAPI / Pydantic should reject an empty question.
    assert response.status_code == 422


# =========================================================
# Run tests manually
# =========================================================

if __name__ == "__main__":

    tests = [
        (
            "Root endpoint",
            test_root
        ),
        (
            "Health endpoint",
            test_health
        ),
        (
            "Relevant retrieval",
            test_relevant_question
        ),
        (
            "Metadata retrieval",
            test_metadata_list_question
        ),
        (
            "Off-topic rejection",
            test_off_topic_question
        ),
        (
            "Input validation",
            test_empty_question_validation
        )
    ]

    passed = 0

    print(
        "\n"
        + "=" * 70
    )

    print(
        "UoM CHATBOT API TESTS"
    )

    print(
        "=" * 70
    )


    for (
        test_name,
        test_function
    ) in tests:

        try:

            test_function()

            print(
                f"PASS - {test_name}"
            )

            passed += 1

        except Exception as error:

            print(
                f"FAIL - {test_name}"
            )

            print(
                f"       {error}"
            )


    print(
        "=" * 70
    )

    print(
        f"RESULT: "
        f"{passed}/{len(tests)} tests passed"
    )

    print(
        "=" * 70
    )