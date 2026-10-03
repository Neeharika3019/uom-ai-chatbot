from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.retriever import retrieve


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="UoM AI Chatbot Retrieval API",
    description=(
        "Backend retrieval API for the University of Mauritius "
        "AI chatbot project."
    ),
    version="1.0.0"
)


# =========================================================
# Request model
# =========================================================

class RetrievalRequest(BaseModel):
    """
    Data expected from the client when asking
    the retrieval system a question.
    """

    question: str = Field(
        ...,
        min_length=1,
        description="Question submitted by the user"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum semantic results to retrieve"
    )


# =========================================================
# Root endpoint
# =========================================================

@app.get("/")
def root():
    """
    Basic API information.
    """

    return {
        "service": "UoM AI Chatbot Retrieval API",
        "version": "1.0.0",
        "status": "running"
    }


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health():
    """
    Used to verify that the API is running.
    """

    return {
        "status": "ok",
        "retriever": "available"
    }


# =========================================================
# Retrieval endpoint
# =========================================================

@app.post("/retrieve")
def retrieve_records(request: RetrievalRequest):
    """
    Retrieve relevant UoM records for a user question.
    """

    try:

        question = request.question.strip()

        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )


        results = retrieve(
            question=question,
            top_k=request.top_k
        )


        # -------------------------------------------------
        # No relevant information found
        # -------------------------------------------------

        if not results:

            return {
                "question": question,
                "has_results": False,
                "result_count": 0,
                "message": (
                    "No relevant UoM information "
                    "was found for this question."
                ),
                "results": []
            }


        # -------------------------------------------------
        # Relevant information found
        # -------------------------------------------------

        return {
            "question": question,
            "has_results": True,
            "result_count": len(results),
            "retrieval_mode": results[0].get(
                "retrieval_mode",
                "unknown"
            ),
            "results": results
        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            f"Retrieval API error: {error}"
        )

        raise HTTPException(
            status_code=500,
            detail="An error occurred during retrieval."
        )