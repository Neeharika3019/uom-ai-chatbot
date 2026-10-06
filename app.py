import os

import httpx
import streamlit as st

from security import (
    validate_input,
    detect_prompt_injection,
    detect_malicious_html
)


# =========================================================
# API configuration
# =========================================================

API_URL = os.getenv(
    "UOM_CHAT_API_URL",
    "http://127.0.0.1:8000/chat"
)

REQUEST_TIMEOUT = 60.0


# =========================================================
# Page configuration
# =========================================================

st.set_page_config(
    page_title="UoM AI Chatbot",
    page_icon="🎓",
    layout="centered"
)


# =========================================================
# Header
# =========================================================

st.title("🎓 University of Mauritius AI Chatbot")

st.write(
    "Welcome! Ask me questions about the University of Mauritius, "
    "including programmes, admissions, academic information, "
    "student services, regulations, faculties and contacts."
)


# =========================================================
# Session state
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# Source display
# =========================================================

def display_sources(sources):

    if not sources:
        return

    with st.expander("Sources"):

        for source in sources:

            title = source.get(
                "title",
                "University of Mauritius source"
            )

            source_url = source.get(
                "source_url",
                ""
            )

            source_file = source.get(
                "source_file",
                ""
            )

            if source_url:

                st.markdown(
                    f"- [{title}]({source_url})"
                )

            elif source_file:

                st.write(
                    f"- {title} ({source_file})"
                )

            else:

                st.write(
                    f"- {title}"
                )


# =========================================================
# Display previous chat messages
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.write(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and
            message.get("sources")
        ):

            display_sources(
                message["sources"]
            )


# =========================================================
# Chat input
# =========================================================

user_question = st.chat_input(
    "Ask a question about UoM..."
)


if user_question:

    # -----------------------------------------------------
    # Save user question
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question
        }
    )


    # -----------------------------------------------------
    # Display user question
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.write(
            user_question
        )


    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------

    is_valid, error_message = validate_input(
        user_question
    )

    sources = []


    if not is_valid:

        response = error_message


    # -----------------------------------------------------
    # Prompt injection protection
    # -----------------------------------------------------

    elif detect_prompt_injection(
        user_question
    ):

        response = (
            "I cannot follow instructions that attempt to override "
            "the chatbot's security rules."
        )


    # -----------------------------------------------------
    # Malicious HTML / XSS protection
    # -----------------------------------------------------

    elif detect_malicious_html(
        user_question
    ):

        response = (
            "Your input contains potentially unsafe HTML "
            "or script content."
        )


    # -----------------------------------------------------
    # Call FastAPI /chat endpoint
    # -----------------------------------------------------

    else:

        try:

            with st.spinner(
                "Searching the University knowledge base..."
            ):

                api_response = httpx.post(
                    API_URL,
                    json={
                        "question": user_question,
                        "top_k": 3
                    },
                    timeout=REQUEST_TIMEOUT
                )


            api_response.raise_for_status()


            data = api_response.json()


            response = data.get(
                "answer",
                "Sorry, I could not generate a response."
            )


            sources = data.get(
                "sources",
                []
            )


        except httpx.ConnectError:

            response = (
                "The chatbot backend is currently unavailable. "
                "Please make sure the API server is running."
            )


        except httpx.TimeoutException:

            response = (
                "The request took too long to complete. "
                "Please try again."
            )


        except httpx.HTTPStatusError:

            response = (
                "The chatbot backend returned an error. "
                "Please try again."
            )


        except Exception:

            response = (
                "Sorry, something went wrong while processing "
                "your question. Please try again."
            )


    # -----------------------------------------------------
    # Save assistant response
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
            "sources": sources
        }
    )


    # -----------------------------------------------------
    # Display assistant response
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        st.write(
            response
        )

        display_sources(
            sources
        )