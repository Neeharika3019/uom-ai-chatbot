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
# Stop words for title matching
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
    "should",
    "uom",
    "university",
    "offer",
    "offers",
    "offered",
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
# Normalise text
# =========================================================

def normalize_text(text):
    """
    Convert text to lowercase and simplify spaces
    and punctuation for easier rule matching.
    """

    text = str(text).lower()

    text = text.replace("-", " ")
    text = text.replace("’", "'")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# Whole word / phrase matching
# =========================================================

def contains_phrase(text, phrase):
    """
    Match a complete word or phrase.

    Example:
    'program' matches 'program'
    but does not match 'programming'.
    """

    pattern = (
        r"\b"
        + re.escape(phrase)
        + r"\b"
    )

    return bool(
        re.search(
            pattern,
            text
        )
    )


# =========================================================
# Tokenise text
# =========================================================

def tokenize(text):
    """
    Convert text into useful lowercase words
    while removing common stop words.
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
# Calculate title match
# =========================================================

def calculate_title_match(question, title):
    """
    Measure overlap between useful words in the
    question and useful words in a record title.
    """

    question_words = tokenize(
        question
    )

    title_words = tokenize(
        title
    )

    if not question_words:
        return 0.0

    matched_words = (
        question_words.intersection(
            title_words
        )
    )

    return (
        len(matched_words)
        / len(question_words)
    )


# =========================================================
# Standardise returned result
# =========================================================

def build_result(
    record,
    score,
    semantic_score=0.0,
    title_match=0.0,
    retrieval_mode="hybrid"
):
    """
    Build a standard result structure.
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
# Get all raw records
# =========================================================

def get_all_records():
    """
    Extract original records from metadata.json.
    """

    return [
        item["record"]
        for item in metadata
    ]


# =========================================================
# Domain intent detection
# =========================================================

def is_uom_domain_question(question):
    """
    Determine whether a question appears to belong
    to the UoM chatbot knowledge domain.
    """

    q = normalize_text(
        question
    )


    # -----------------------------------------------------
    # Strong institutional signals
    # -----------------------------------------------------

    institutional_signals = [
        "uom",
        "university of mauritius",
        "foicdt"
    ]

    if any(
        signal in q
        for signal in institutional_signals
    ):
        return True


    # -----------------------------------------------------
    # Degree / programme terminology
    # -----------------------------------------------------

    degree_signals = [
        "programme",
        "programmes",
        "program",
        "programs",
        "undergraduate",
        "undergraduates",
        "postgraduate",
        "postgraduates",
        "bachelor",
        "bachelors",
        "bachelor's",
        "master",
        "masters",
        "master's",
        "bsc",
        "msc",
        "degree",
        "degrees"
    ]

    if any(
        contains_phrase(
            q,
            signal
        )
        for signal in degree_signals
    ):
        return True


    # -----------------------------------------------------
    # Faculty / department signals
    # -----------------------------------------------------

    faculty_signals = [
        "dean",
        "department",
        "departments",
        "digital technologies",
        "software and information systems",
        "information and communication technology"
    ]

    if any(
        signal in q
        for signal in faculty_signals
    ):
        return True


    # -----------------------------------------------------
    # Admission / eligibility signals
    # -----------------------------------------------------

    admission_signals = [
        "eligibility",
        "eligible",
        "admission",
        "admissions",
        "applicant",
        "applicants"
    ]

    if any(
        contains_phrase(
            q,
            signal
        )
        for signal in admission_signals
    ):
        return True


    # -----------------------------------------------------
    # Documents + application context
    # -----------------------------------------------------

    document_signal = any(
        contains_phrase(
            q,
            signal
        )
        for signal in [
            "document",
            "documents",
            "paperwork"
        ]
    )

    application_signal = any(
        contains_phrase(
            q,
            signal
        )
        for signal in [
            "apply",
            "applying",
            "application"
        ]
    )

    if (
        document_signal
        and application_signal
    ):
        return True


    # -----------------------------------------------------
    # Fee / payment signals
    # -----------------------------------------------------

    financial_signals = [
        "fee",
        "fees",
        "payment",
        "payments",
        "pay",
        "paying",
        "settle",
        "administrative charge",
        "administrative charges",
        "administrative fee",
        "administrative fees"
    ]

    if any(
        contains_phrase(
            q,
            signal
        )
        for signal in financial_signals
    ):
        return True


    return False


# =========================================================
# Detect undergraduate request
# =========================================================

def detect_undergraduate_request(question):
    """
    Recognise different ways of referring to
    undergraduate programmes.
    """

    q = normalize_text(
        question
    )

    undergraduate_terms = [
        "undergraduate",
        "undergraduates",
        "bachelor",
        "bachelors",
        "bachelor's",
        "bsc"
    ]

    return any(
        contains_phrase(
            q,
            term
        )
        for term in undergraduate_terms
    )


# =========================================================
# Detect postgraduate request
# =========================================================

def detect_postgraduate_request(question):
    """
    Recognise different ways of referring to
    postgraduate programmes.
    """

    q = normalize_text(
        question
    )

    postgraduate_terms = [
        "postgraduate",
        "postgraduates",
        "master",
        "masters",
        "master's",
        "msc"
    ]

    return any(
        contains_phrase(
            q,
            term
        )
        for term in postgraduate_terms
    )


# =========================================================
# Detect list / group query
# =========================================================

def detect_list_intent(question):
    """
    Detect whether the user wants several records
    rather than information about one specific programme.

    Examples:

    "Show me all bachelor's programmes."
        -> True

    "List every undergraduate degree."
        -> True

    "Which undergraduate programmes are three years?"
        -> True

    "Give me a description of the BSc Software
    Engineering programme."
        -> False
    """

    q = normalize_text(
        question
    )


    # -----------------------------------------------------
    # Detect specific named degree reference
    # -----------------------------------------------------

    specific_degree_reference = (
        contains_phrase(
            q,
            "bsc"
        )
        or contains_phrase(
            q,
            "msc"
        )
    )


    # -----------------------------------------------------
    # Detect request for information about one programme
    # -----------------------------------------------------

    specific_information_signals = [
        "description",
        "describe",
        "tell me about",
        "information about",
        "how long is",
        "duration of"
    ]

    specific_information_request = any(
        signal in q
        for signal in specific_information_signals
    )


    # -----------------------------------------------------
    # Explicit multi-record wording
    # -----------------------------------------------------

    explicit_group_signals = [
        "show me all",
        "list all",
        "list the",
        "list every",
        "all the",
        "which programmes",
        "which programs",
        "which degrees",
        "which courses"
    ]

    explicit_group_request = any(
        signal in q
        for signal in explicit_group_signals
    )


    # -----------------------------------------------------
    # Specific BSc / MSc request should remain hybrid
    # unless the user explicitly asks for a group/list.
    # -----------------------------------------------------

    if (
        specific_degree_reference
        and specific_information_request
        and not explicit_group_request
    ):
        return False


    # -----------------------------------------------------
    # Explicit list wording
    # -----------------------------------------------------

    explicit_list_signals = [
        "show me all",
        "list all",
        "list the",
        "list every",
        "all the",
        "what are the",
        "what are",
        "which programmes",
        "which programs",
        "which degrees",
        "which courses",
        "different departments",
        "each department",
        "every undergraduate",
        "every postgraduate",
        "every degree",
        "every programme",
        "every program"
    ]

    if any(
        signal in q
        for signal in explicit_list_signals
    ):
        return True


    # -----------------------------------------------------
    # General "every" wording
    #
    # Examples:
    # "every undergraduate degree"
    # "every postgraduate programme"
    # -----------------------------------------------------

    if contains_phrase(
        q,
        "every"
    ):

        if (
            detect_undergraduate_request(
                question
            )
            or detect_postgraduate_request(
                question
            )
            or contains_phrase(
                q,
                "degree"
            )
            or contains_phrase(
                q,
                "degrees"
            )
            or contains_phrase(
                q,
                "programme"
            )
            or contains_phrase(
                q,
                "programmes"
            )
            or contains_phrase(
                q,
                "program"
            )
            or contains_phrase(
                q,
                "programs"
            )
        ):
            return True


    # -----------------------------------------------------
    # Plural programme wording
    # -----------------------------------------------------

    plural_terms = [
        "programmes",
        "programs",
        "degrees",
        "courses",
        "undergraduates",
        "postgraduates"
    ]

    if any(
        contains_phrase(
            q,
            term
        )
        for term in plural_terms
    ):
        return True


    # -----------------------------------------------------
    # Master's list wording
    # -----------------------------------------------------

    if (
        contains_phrase(
            q,
            "masters"
        )
        and (
            "available" in q
            or "which" in q
            or "what are" in q
        )
    ):
        return True


    return False


# =========================================================
# Identify undergraduate record
# =========================================================

def is_undergraduate_record(record):
    """
    Determine whether a record represents an
    undergraduate programme.
    """

    record_id = normalize_text(
        record.get(
            "ID",
            ""
        )
    )

    subcategory = normalize_text(
        record.get(
            "Subcategory",
            ""
        )
    )

    title = normalize_text(
        record.get(
            "Title",
            ""
        )
    )

    if not record_id.startswith(
        "prog_"
    ):
        return False

    return (
        "undergraduate" in subcategory
        or title.startswith("bsc")
    )


# =========================================================
# Identify postgraduate record
# =========================================================

def is_postgraduate_record(record):
    """
    Determine whether a record represents a
    postgraduate programme.
    """

    record_id = normalize_text(
        record.get(
            "ID",
            ""
        )
    )

    subcategory = normalize_text(
        record.get(
            "Subcategory",
            ""
        )
    )

    title = normalize_text(
        record.get(
            "Title",
            ""
        )
    )

    if not record_id.startswith(
        "prog_"
    ):
        return False

    return (
        "postgraduate" in subcategory
        or title.startswith("msc")
    )


# =========================================================
# Detect three-year programme
# =========================================================

def is_three_year_record(record):
    """
    Check whether a programme has a three-year
    duration.
    """

    duration = normalize_text(
        record.get(
            "Duration",
            ""
        )
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
    Handle structured and multi-answer queries.
    """

    q = normalize_text(
        question
    )

    records = get_all_records()


    undergraduate_request = (
        detect_undergraduate_request(
            question
        )
    )

    postgraduate_request = (
        detect_postgraduate_request(
            question
        )
    )

    list_intent = detect_list_intent(
        question
    )


    # -----------------------------------------------------
    # Department list query
    # -----------------------------------------------------

    department_list_request = (
        (
            "departments" in q
            or "different departments" in q
            or "each department" in q
        )
        and (
            "foicdt" in q
            or "heads" in q
            or "head" in q
        )
    )


    if department_list_request:

        department_records = [
            record
            for record in records
            if (
                str(
                    record.get(
                        "ID",
                        ""
                    )
                ).startswith("FAC_")
                and
                "department"
                in normalize_text(
                    record.get(
                        "Title",
                        ""
                    )
                )
            )
        ]


        department_records.sort(
            key=lambda record:
            record.get(
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
            for record
            in department_records
        ]


    # -----------------------------------------------------
    # Undergraduate list / filter
    # -----------------------------------------------------

    if (
        undergraduate_request
        and list_intent
    ):

        programme_records = [
            record
            for record in records
            if is_undergraduate_record(
                record
            )
        ]


        asks_for_three_years = (
            bool(
                re.search(
                    r"\b3\s*(year|years|yr|yrs)\b",
                    q
                )
            )
            or "three years" in q
            or "three year" in q
        )


        if asks_for_three_years:

            programme_records = [
                record
                for record in programme_records
                if is_three_year_record(
                    record
                )
            ]


        programme_records.sort(
            key=lambda record:
            record.get(
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
            for record
            in programme_records
        ]


    # -----------------------------------------------------
    # Postgraduate list
    # -----------------------------------------------------

    if (
        postgraduate_request
        and list_intent
    ):

        programme_records = [
            record
            for record in records
            if is_postgraduate_record(
                record
            )
        ]


        programme_records.sort(
            key=lambda record:
            record.get(
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
            for record
            in programme_records
        ]


    return None


# =========================================================
# Intent-based reranking
# =========================================================

def calculate_intent_boost(
    question,
    record
):
    """
    Apply small boosts for clearly identifiable
    query intents.
    """

    q = normalize_text(
        question
    )

    record_id = str(
        record.get(
            "ID",
            ""
        )
    )


    boost = 0.0


    # -----------------------------------------------------
    # Administrative fees
    # -----------------------------------------------------

    administrative_signal = (
        "administrative" in q
    )

    fee_signal = any(
        contains_phrase(
            q,
            term
        )
        for term in [
            "fee",
            "fees",
            "charge",
            "charges",
            "cost",
            "costs"
        ]
    )


    if (
        record_id == "ADM_003"
        and administrative_signal
        and fee_signal
    ):
        boost += 0.20


    # -----------------------------------------------------
    # Payment methods
    # -----------------------------------------------------

    payment_signal = any(
        contains_phrase(
            q,
            term
        )
        for term in [
            "payment",
            "payments",
            "pay",
            "paying",
            "settle",
            "means"
        ]
    )


    if (
        record_id == "ADM_004"
        and payment_signal
    ):
        boost += 0.12


    # -----------------------------------------------------
    # Required documents
    # -----------------------------------------------------

    document_signal = any(
        contains_phrase(
            q,
            term
        )
        for term in [
            "document",
            "documents",
            "paperwork"
        ]
    )


    if (
        record_id == "ADM_001"
        and document_signal
    ):
        boost += 0.12


    # -----------------------------------------------------
    # Eligibility
    # -----------------------------------------------------

    eligibility_signal = any(
        contains_phrase(
            q,
            term
        )
        for term in [
            "eligible",
            "eligibility",
            "qualifications",
            "qualify"
        ]
    )


    if (
        record_id == "ADM_002"
        and eligibility_signal
    ):
        boost += 0.12


    # -----------------------------------------------------
    # Cybersecurity programme intent
    # -----------------------------------------------------

    cybersecurity_signal = any(
        contains_phrase(
            q,
            term
        )
        for term in [
            "cybersecurity",
            "cyber security",
            "security",
            "protect",
            "protecting",
            "hackers",
            "malware",
            "cyber attack",
            "cyber attacks"
        ]
    )


    if (
        record_id in [
            "PROG_006",
            "PROG_007"
        ]
        and cybersecurity_signal
    ):
        boost += 0.08


    return boost


# =========================================================
# Main retrieval function
# =========================================================

def retrieve(
    question,
    top_k=3,
    min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE
):
    """
    Main hybrid retrieval function.

    Stages:
    1. Validate question
    2. Check UoM domain intent
    3. Try metadata retrieval
    4. Use FAISS semantic search
    5. Apply title and intent reranking
    6. Reject weak semantic matches
    """

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question:
        return []


    question = question.strip()


    if not question:
        return []


    # -----------------------------------------------------
    # Domain intent check
    # -----------------------------------------------------

    if not is_uom_domain_question(
        question
    ):
        return []


    # -----------------------------------------------------
    # Metadata retrieval
    # -----------------------------------------------------

    metadata_results = metadata_retrieve(
        question
    )


    if metadata_results is not None:
        return metadata_results


    # -----------------------------------------------------
    # Encode question
    # -----------------------------------------------------

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )


    # -----------------------------------------------------
    # Candidate retrieval
    # -----------------------------------------------------

    candidate_k = min(
        max(
            top_k * 2,
            5
        ),
        index.ntotal
    )


    semantic_scores, indices = (
        index.search(
            question_embedding,
            candidate_k
        )
    )


    results = []


    normalized_question = normalize_text(
        question
    )


    # -----------------------------------------------------
    # Detect explicit Top-Up wording
    # -----------------------------------------------------

    question_mentions_top_up = bool(
        re.search(
            r"\btop\s*up\b",
            normalized_question
        )
    )


    # -----------------------------------------------------
    # Process candidates
    # -----------------------------------------------------

    for (
        semantic_score,
        index_position
    ) in zip(
        semantic_scores[0],
        indices[0]
    ):

        if index_position == -1:
            continue


        semantic_score = float(
            semantic_score
        )


        # -------------------------------------------------
        # Semantic relevance threshold
        # -------------------------------------------------

        if (
            semantic_score
            < min_semantic_score
        ):
            continue


        item = metadata[
            index_position
        ]

        record = item[
            "record"
        ]


        title = record.get(
            "Title",
            ""
        )

        normalized_title = normalize_text(
            title
        )


        # -------------------------------------------------
        # Title score
        # -------------------------------------------------

        title_match = calculate_title_match(
            question,
            title
        )


        # -------------------------------------------------
        # Intent score
        # -------------------------------------------------

        intent_boost = calculate_intent_boost(
            question,
            record
        )


        # -------------------------------------------------
        # Final hybrid score
        # -------------------------------------------------

        final_score = (
            semantic_score
            + (
                0.25
                * title_match
            )
            + intent_boost
        )


        # -------------------------------------------------
        # Top-Up disambiguation
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


        results.append(
            result
        )


    # -----------------------------------------------------
    # Sort by final score
    # -----------------------------------------------------

    results.sort(
        key=lambda result:
        result["score"],
        reverse=True
    )


    # -----------------------------------------------------
    # Return final results
    # -----------------------------------------------------

    return results[
        :top_k
    ]


# =========================================================
# Manual testing
# =========================================================

if __name__ == "__main__":

    test_questions = [

        (
            "List every undergraduate degree "
            "available under FOICDT."
        ),

        (
            "Can I have a description about the "
            "BSc (Hons) Software Engineering programmes?"
        ),

        (
            "I want to study a degree focused on "
            "protecting computer systems and networks. "
            "What undergraduate programme does FOICDT offer?"
        ),

        (
            "Show me all the bachelor's programmes "
            "offered by FOICDT."
        ),

        (
            "Which FOICDT bachelor's degrees take "
            "three years to complete?"
        ),

        (
            "Which Master's programmes can I study "
            "under FOICDT?"
        ),

        (
            "I would like information about the yearly "
            "administrative charges for Mauritian students."
        ),

        (
            "How can I protect my personal laptop "
            "from malware and hackers?"
        ),

        (
            "Can you explain what artificial "
            "intelligence is?"
        ),

        (
            "What programming language should a "
            "beginner learn first?"
        ),

        (
            "How do I apply for a passport in Mauritius?"
        )
    ]


    for question in test_questions:

        print(
            "\n"
            + "=" * 75
        )


        print(
            f"QUESTION: {question}"
        )


        results = retrieve(
            question,
            top_k=3
        )


        if not results:

            print(
                "RETRIEVAL MODE: NONE"
            )

            print(
                "No relevant UoM information found."
            )

            continue


        print(
            "RETRIEVAL MODE: "
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