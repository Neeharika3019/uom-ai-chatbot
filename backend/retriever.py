import json
import re

import faiss
from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================

INDEX_FILE = "vector_db/uom_index.faiss"
METADATA_FILE = "vector_db/metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_MIN_SEMANTIC_SCORE = 0.20


# =========================================================
# Load embedding model
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer(MODEL_NAME)


# =========================================================
# Load FAISS index
# =========================================================

print("Loading FAISS index...")

index = faiss.read_index(INDEX_FILE)


# =========================================================
# Load metadata
# =========================================================

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as file:
    metadata = json.load(file)


# =========================================================
# Stop words used for title matching
# =========================================================

STOP_WORDS = {
    "what",
    "which",
    "who",
    "how",
    "when",
    "where",
    "why",
    "is",
    "are",
    "was",
    "were",
    "the",
    "a",
    "an",
    "of",
    "in",
    "at",
    "to",
    "for",
    "from",
    "does",
    "do",
    "did",
    "can",
    "could",
    "would",
    "uom",
    "university",
    "offer",
    "offers",
    "available",
    "programme",
    "programmes",
    "program",
    "programs",
    "course",
    "courses",
    "tell",
    "me",
    "about"
}


# =========================================================
# Text normalisation
# =========================================================

def normalize_text(text):
    """
    Normalise text for easier rule matching.
    """

    text = str(text).lower()

    text = text.replace("-", " ")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# Tokenisation
# =========================================================

def tokenize(text):
    """
    Convert text into useful lowercase words while
    removing common stop words.
    """

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        str(text).lower()
    )

    useful_words = {
        word
        for word in words
        if word not in STOP_WORDS
    }

    return useful_words


# =========================================================
# Title-match score
# =========================================================

def calculate_title_match(question, title):
    """
    Compare important words from the user's question
    against words in the record title.

    Returns a value between 0 and 1.
    """

    question_words = tokenize(question)
    title_words = tokenize(title)

    if not question_words:
        return 0.0

    matched_words = question_words.intersection(
        title_words
    )

    return (
        len(matched_words)
        / len(question_words)
    )


# =========================================================
# Convert record to standard result format
# =========================================================

def build_result(
    record,
    score,
    semantic_score=0.0,
    title_match=0.0,
    retrieval_mode="hybrid"
):
    """
    Return all retrieved records using the same
    structure regardless of retrieval method.
    """

    description = (
        record.get("Programme Description")
        or record.get("Description", "")
    )

    return {
        "id": record.get(
            "ID",
            ""
        ),

        "title": record.get(
            "Title",
            ""
        ),

        "category": record.get(
            "Category",
            ""
        ),

        "subcategory": record.get(
            "Subcategory",
            ""
        ),

        "faculty": record.get(
            "Faculty",
            ""
        ),

        "duration": record.get(
            "Duration",
            ""
        ),

        "academic_year": record.get(
            "Academic Year",
            ""
        ),

        "applicant_type": record.get(
            "Applicant Type",
            ""
        ),

        "description": description,

        "contact_email": record.get(
            "Contact Email",
            ""
        ),

        "source_url": record.get(
            "Source URL",
            ""
        ),

        "semantic_score": float(
            semantic_score
        ),

        "title_match": float(
            title_match
        ),

        "score": float(
            score
        ),

        "retrieval_mode": retrieval_mode
    }


# =========================================================
# Get all records
# =========================================================

def get_all_records():
    """
    Extract raw records from metadata.json.
    """

    return [
        item["record"]
        for item in metadata
    ]


# =========================================================
# Programme-level helpers
# =========================================================

def is_undergraduate_record(record):
    """
    Determine whether a record represents an
    undergraduate programme.
    """

    record_id = normalize_text(
        record.get("ID", "")
    )

    subcategory = normalize_text(
        record.get("Subcategory", "")
    )

    title = normalize_text(
        record.get("Title", "")
    )

    if not record_id.startswith("prog_"):
        return False

    return (
        "undergraduate" in subcategory
        or title.startswith("bsc")
    )


def is_postgraduate_record(record):
    """
    Determine whether a record represents a
    postgraduate programme.
    """

    record_id = normalize_text(
        record.get("ID", "")
    )

    subcategory = normalize_text(
        record.get("Subcategory", "")
    )

    title = normalize_text(
        record.get("Title", "")
    )

    if not record_id.startswith("prog_"):
        return False

    return (
        "postgraduate" in subcategory
        or title.startswith("msc")
    )


def is_three_year_record(record):
    """
    Check whether the programme duration represents
    approximately 3 years.
    """

    duration = normalize_text(
        record.get("Duration", "")
    )

    return bool(
        re.search(
            r"\b3\s*(year|years|yr|yrs)\b",
            duration
        )
    )


# =========================================================
# Metadata-aware retrieval
# =========================================================

def metadata_retrieve(question):
    """
    Detect structured/list-style questions that are
    better answered using metadata rather than only
    vector similarity.

    Returns:
        list of matching results if metadata retrieval
        should be used.

        None if normal semantic retrieval should be used.
    """

    q = normalize_text(question)

    records = get_all_records()

    # -----------------------------------------------------
    # Detect undergraduate / postgraduate concepts
    # -----------------------------------------------------

    undergraduate_request = (
        "undergraduate" in q
        or "undergraduates" in q
    )

    postgraduate_request = (
        "postgraduate" in q
        or "postgraduates" in q
        or bool(
            re.search(
                r"\bmasters?\b",
                q
            )
        )
        or bool(
            re.search(
                r"\bmaster's\b",
                q
            )
        )
    )

    # -----------------------------------------------------
    # Detect list/group intent
    # -----------------------------------------------------

    list_intent = any(
        phrase in q
        for phrase in [
            "what are",
            "which ",
            "list",
            "lists",
            "available",
            "all ",
            "each "
        ]
    )

    # Questions such as:
    # "How long are the postgraduate programmes?"
    # are also group queries.
    programme_group_request = (
        (
            undergraduate_request
            or postgraduate_request
        )
        and (
            "programme" in q
            or "programmes" in q
            or "program" in q
            or "programs" in q
        )
    )

    # -----------------------------------------------------
    # Department list query
    #
    # Example:
    # "What departments are under FOICDT and who heads
    # each department?"
    # -----------------------------------------------------

    department_list_request = (
        (
            "departments" in q
            or (
                "department" in q
                and (
                    "each" in q
                    or "all" in q
                    or "list" in q
                )
            )
        )
        and "foicdt" in q
    )

    if department_list_request:

        department_records = [
            record
            for record in records
            if (
                str(
                    record.get("ID", "")
                ).startswith("FAC_")
                and "department" in normalize_text(
                    record.get("Title", "")
                )
            )
        ]

        department_records.sort(
            key=lambda record: record.get(
                "ID",
                ""
            )
        )

        return [
            build_result(
                record=record,
                score=1.0,
                retrieval_mode="metadata"
            )
            for record in department_records
        ]

    # -----------------------------------------------------
    # Undergraduate programme list
    # -----------------------------------------------------

    if (
        undergraduate_request
        and (
            list_intent
            or programme_group_request
        )
    ):

        programme_records = [
            record
            for record in records
            if is_undergraduate_record(record)
        ]

        # ---------------------------------------------
        # Optional duration filter
        #
        # Example:
        # "Which undergraduate programmes are 3 years?"
        # ---------------------------------------------

        asks_for_three_years = bool(
            re.search(
                r"\b3\s*(year|years|yr|yrs)\b",
                q
            )
        )

        if asks_for_three_years:

            programme_records = [
                record
                for record in programme_records
                if is_three_year_record(record)
            ]

        programme_records.sort(
            key=lambda record: record.get(
                "ID",
                ""
            )
        )

        return [
            build_result(
                record=record,
                score=1.0,
                retrieval_mode="metadata"
            )
            for record in programme_records
        ]

    # -----------------------------------------------------
    # Postgraduate programme list
    # -----------------------------------------------------

    if (
        postgraduate_request
        and (
            list_intent
            or programme_group_request
        )
    ):

        programme_records = [
            record
            for record in records
            if is_postgraduate_record(record)
        ]

        programme_records.sort(
            key=lambda record: record.get(
                "ID",
                ""
            )
        )

        return [
            build_result(
                record=record,
                score=1.0,
                retrieval_mode="metadata"
            )
            for record in programme_records
        ]

    # -----------------------------------------------------
    # No metadata rule matched
    # -----------------------------------------------------

    return None


# =========================================================
# Main retrieval function
# =========================================================

def retrieve(
    question,
    top_k=3,
    min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE
):
    """
    Hybrid retrieval system.

    Retrieval modes:

    1. Metadata retrieval
       Used for list/filter questions.

    2. FAISS semantic retrieval + title reranking
       Used for specific semantic questions.

    3. Off-topic rejection
       Weak semantic matches below the relevance
       threshold are removed.
    """

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question or not question.strip():
        return []

    # -----------------------------------------------------
    # STEP 1:
    # Try metadata-aware retrieval first
    # -----------------------------------------------------

    metadata_results = metadata_retrieve(
        question
    )

    if metadata_results is not None:

        return metadata_results

    # -----------------------------------------------------
    # STEP 2:
    # Semantic retrieval using FAISS
    # -----------------------------------------------------

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    candidate_k = min(
        max(top_k * 2, 5),
        index.ntotal
    )

    semantic_scores, indices = index.search(
        question_embedding,
        candidate_k
    )

    results = []

    normalized_question = normalize_text(
        question
    )

    # -----------------------------------------------------
    # Does the user explicitly mention Top Up?
    # -----------------------------------------------------

    question_mentions_top_up = bool(
        re.search(
            r"\btop\s*up\b",
            normalized_question
        )
    )

    # -----------------------------------------------------
    # Process semantic candidates
    # -----------------------------------------------------

    for semantic_score, index_position in zip(
        semantic_scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue

        semantic_score = float(
            semantic_score
        )

        # -------------------------------------------------
        # Off-topic / relevance threshold
        # -------------------------------------------------

        if semantic_score < min_semantic_score:
            continue

        item = metadata[index_position]

        record = item["record"]

        title = record.get(
            "Title",
            ""
        )

        normalized_title = normalize_text(
            title
        )

        # -------------------------------------------------
        # Calculate title match
        # -------------------------------------------------

        title_match = calculate_title_match(
            question,
            title
        )

        # -------------------------------------------------
        # Base hybrid score
        # -------------------------------------------------

        final_score = (
            semantic_score
            + (0.25 * title_match)
        )

        # -------------------------------------------------
        # Top-Up disambiguation
        #
        # If user asks simply for BSc Cybersecurity,
        # slightly prefer the standard programme.
        #
        # If user explicitly asks for Top Up,
        # slightly boost the Top Up record.
        # -------------------------------------------------

        title_is_top_up = bool(
            re.search(
                r"\btop\s*up\b",
                normalized_title
            )
        )

        if title_is_top_up:

            if question_mentions_top_up:

                final_score += 0.08

            else:

                final_score -= 0.08

        # -------------------------------------------------
        # Build result
        # -------------------------------------------------

        result = build_result(
            record=record,
            semantic_score=semantic_score,
            title_match=title_match,
            score=final_score,
            retrieval_mode="hybrid"
        )

        results.append(result)

    # -----------------------------------------------------
    # Sort by final score
    # -----------------------------------------------------

    results.sort(
        key=lambda result: result["score"],
        reverse=True
    )

    return results[:top_k]


# =========================================================
# Manual testing
# =========================================================

if __name__ == "__main__":

    test_questions = [
        "How long is the BSc Cybersecurity programme?",
        "What is the Cybersecurity Top Up programme?",
        "What undergraduate programmes are available under FOICDT?",
        "Which undergraduate programmes are 3 years?",
        "What Masters are available at UoM in the IT sector?",
        "How long are the postgraduate programmes under FOICDT?",
        "What departments are present under FOICDT and who heads each one?",
        "Who is the Dean of FOICDT?",
        "What is the weather tomorrow?"
    ]

    for question in test_questions:

        print("\n" + "=" * 75)

        print(
            f"QUESTION: {question}"
        )

        results = retrieve(
            question,
            top_k=3
        )

        if not results:

            print(
                "No relevant UoM information found."
            )

            continue

        print(
            f"RETRIEVAL MODE: "
            f"{results[0]['retrieval_mode']}"
        )

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