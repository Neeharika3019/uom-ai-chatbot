from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from backend.retriever import retrieve
from backend.chat_engine import answer_question


# =========================================================
# FastAPI application
# =========================================================

app = FastAPI(
    title="UoM AI Chatbot API",
    description=(
        "Retrieval and RAG chatbot API for the "
        "University of Mauritius AI Chatbot project."
    ),
    version="2.0.0"
)


# =========================================================
# Request models
# =========================================================

class RetrievalRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Question to search in the UoM knowledge base."
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of retrieval results."
    )


class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        description="Student question for the UoM AI chatbot."
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description=(
            "Maximum number of knowledge records supplied "
            "to the chatbot."
        )
    )


# =========================================================
# Root endpoint
# =========================================================

@app.get("/")
def root():

    return {
        "service": "UoM AI Chatbot API",
        "version": "2.0.0",
        "status": "running"
    }


# =========================================================
# Health endpoint
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "retriever": "available",
        "chat_engine": "available"
    }


# =========================================================
# Retrieval endpoint
# =========================================================

@app.post("/retrieve")
def retrieve_records(
    request: RetrievalRequest
):

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
        # No relevant information
        # -------------------------------------------------

        if not results:

            return {
                "question": question,
                "has_results": False,
                "result_count": 0,
                "message": (
                    "No relevant UoM information was "
                    "found for this question."
                ),
                "results": []
            }


        # -------------------------------------------------
        # Successful retrieval
        # -------------------------------------------------

        return {
            "question": question,
            "has_results": True,
            "result_count": len(
                results
            ),
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
            "Retrieval API error:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail="Internal retrieval error."
        )


# =========================================================
# Chat endpoint
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    try:

        question = request.question.strip()


        if not question:

            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )


        response = answer_question(
            question=question,
            top_k=request.top_k
        )


        return response


    except HTTPException:

        raise


    except Exception as error:

        print(
            "Chat API error:",
            str(error)
        )

        raise HTTPException(
            status_code=500,
            detail="Internal chatbot error."
        )