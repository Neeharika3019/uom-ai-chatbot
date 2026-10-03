import streamlit as st

from security import (
    validate_input,
    detect_prompt_injection,
    detect_malicious_html
)

# Page configuration
st.set_page_config(
    page_title="UoM AI Chatbot",
    page_icon="🎓",
    layout="centered"
)

# Title
st.title("🎓 University of Mauritius AI Chatbot")

# Introduction
st.write(
    "Welcome! Ask me questions about the University of Mauritius, "
    "including programmes, admissions, regulations, faculties and contacts."
)

# Create chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
user_question = st.chat_input("Ask a question about UoM...")

if user_question:

    # Save and display user's question
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.write(user_question)

    # Validate input
    is_valid, error_message = validate_input(user_question)

    if not is_valid:
        response = error_message

    # Check for prompt injection
    elif detect_prompt_injection(user_question):
        response = (
            "I cannot follow instructions that attempt to override "
            "the chatbot's security rules."
        )

    # Check for malicious HTML / XSS
    elif detect_malicious_html(user_question):
        response = (
            "Your input contains potentially unsafe HTML or script content."
        )

    # Process normal user question
    else:
        try:
            # Temporary response for now.
            # Later, the FAISS retriever and LLM will be connected here.
            response = (
                "This is a temporary chatbot response. "
                "The AI and retrieval system will be connected later."
            )

        except Exception:
            response = (
                "Sorry, something went wrong while processing your question. "
                "Please try again."
            )

    # Save chatbot response
    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    # Display chatbot response
    with st.chat_message("assistant"):
        st.write(response)