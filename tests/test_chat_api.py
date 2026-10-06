import os
import sys
from unittest.mock import patch

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


client = TestClient(app)


# =========================================================
# Test 1: Valid chat request
# =========================================================

def test_chat_valid_question():

    fake_response = {
        "question": "When do Semester 1 examinations start and end?",
        "has_results": True,
        "answer": (
            "Semester 1 examinations start on Tuesday "
            "03 November 2026 and end on Saturday "
            "28 November 2026."
        ),
        "result_count": 3,
        "retrieval_mode": "hybrid",
        "llm_used": True,
        "sources": [
            {
                "id": "ACAD_005",
                "title": "Semester 1 Examination Period",
                "source_url": "",
                "source_file": "academic-information.txt"
            }
        ],
        "retrieved_records": []
    }


    with patch(
        "backend.api.answer_question",
        return_value=fake_response
    ):

        response = client.post(
            "/chat",
            json={
                "question": (
                    "When do Semester 1 examinations "
                    "start and end?"
                ),
                "top_k": 3
            }
        )


    assert response.status_code == 200


    data = response.json()


    assert data["has_results"] is True

    assert data["llm_used"] is True

    assert data["retrieval_mode"] == "hybrid"

    assert data["result_count"] == 3

    assert "03 November 2026" in data["answer"]

    assert data["sources"][0]["id"] == "ACAD_005"


# =========================================================
# Test 2: Unsupported question
# =========================================================

def test_chat_unsupported_question():

    fake_response = {
        "question": "Who is the Prime Minister of Australia?",
        "has_results": False,
        "answer": (
            "I could not find sufficient information about this "
            "in the University of Mauritius knowledge base."
        ),
        "result_count": 0,
        "retrieval_mode": "none",
        "llm_used": False,
        "sources": [],
        "retrieved_records": []
    }


    with patch(
        "backend.api.answer_question",
        return_value=fake_response
    ):

        response = client.post(
            "/chat",
            json={
                "question": (
                    "Who is the Prime Minister of Australia?"
                ),
                "top_k": 3
            }
        )


    assert response.status_code == 200


    data = response.json()


    assert data["has_results"] is False

    assert data["result_count"] == 0

    assert data["retrieval_mode"] == "none"

    assert data["llm_used"] is False

    assert data["sources"] == []


# =========================================================
# Test 3: Missing question
# =========================================================

def test_chat_missing_question():

    response = client.post(
        "/chat",
        json={
            "top_k": 3
        }
    )


    assert response.status_code == 422


# =========================================================
# Test 4: Empty question
# =========================================================

def test_chat_empty_question():

    response = client.post(
        "/chat",
        json={
            "question": "",
            "top_k": 3
        }
    )


    assert response.status_code == 422


# =========================================================
# Test 5: Whitespace-only question
# =========================================================

def test_chat_whitespace_question():

    response = client.post(
        "/chat",
        json={
            "question": "   ",
            "top_k": 3
        }
    )


    assert response.status_code == 400


# =========================================================
# Test 6: top_k below allowed range
# =========================================================

def test_chat_invalid_low_top_k():

    response = client.post(
        "/chat",
        json={
            "question": "What programmes are available?",
            "top_k": 0
        }
    )


    assert response.status_code == 422


# =========================================================
# Test 7: top_k above allowed range
# =========================================================

def test_chat_invalid_high_top_k():

    response = client.post(
        "/chat",
        json={
            "question": "What programmes are available?",
            "top_k": 11
        }
    )


    assert response.status_code == 422


# =========================================================
# Test 8: Confirm top_k is passed to chat engine
# =========================================================

def test_chat_passes_top_k():

    fake_response = {
        "question": "What cybersecurity programmes are available?",
        "has_results": True,
        "answer": "Test answer",
        "result_count": 5,
        "retrieval_mode": "hybrid",
        "llm_used": True,
        "sources": [],
        "retrieved_records": []
    }


    with patch(
        "backend.api.answer_question",
        return_value=fake_response
    ) as mocked_answer:

        response = client.post(
            "/chat",
            json={
                "question": (
                    "What cybersecurity programmes "
                    "are available?"
                ),
                "top_k": 5
            }
        )


    assert response.status_code == 200


    mocked_answer.assert_called_once_with(
        question=(
            "What cybersecurity programmes "
            "are available?"
        ),
        top_k=5
    )


# =========================================================
# Manual runner
# =========================================================

if __name__ == "__main__":

    tests = [
        test_chat_valid_question,
        test_chat_unsupported_question,
        test_chat_missing_question,
        test_chat_empty_question,
        test_chat_whitespace_question,
        test_chat_invalid_low_top_k,
        test_chat_invalid_high_top_k,
        test_chat_passes_top_k
    ]


    passed = 0


    for test in tests:

        try:

            test()

            print(
                f"PASS: {test.__name__}"
            )

            passed += 1


        except Exception as error:

            print(
                f"FAIL: {test.__name__}"
            )

            print(
                f"      {error}"
            )


    print(
        "\n"
        + "=" * 60
    )

    print(
        f"CHAT API TEST RESULT: "
        f"{passed}/{len(tests)} passed"
    )

    print(
        "=" * 60
    )