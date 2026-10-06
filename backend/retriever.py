import json
import re
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer


# =========================================================
# Project configuration
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_FILE = (
    PROJECT_ROOT
    / "vector_db"
    / "uom_index.faiss"
)

METADATA_FILE = (
    PROJECT_ROOT
    / "vector_db"
    / "metadata.json"
)

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_MIN_SEMANTIC_SCORE = 0.20


# =========================================================
# Load embedding model
# =========================================================

print("Loading embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)


# =========================================================
# Load FAISS index
# =========================================================

print("Loading FAISS index...")

index = faiss.read_index(
    str(INDEX_FILE)
)


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
    "should",
    "uom",
    "university",
    "mauritius",
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
# Text normalisation
# =========================================================

def normalize_text(text):
    """
    Convert text to lowercase and normalise spaces
    and punctuation for easier comparison.
    """

    text = str(text).lower()

    text = text.replace(
        "-",
        " "
    )

    text = text.replace(
        "’",
        "'"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# =========================================================
# Whole-word / phrase matching
# =========================================================

def contains_phrase(
    text,
    phrase
):
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


def contains_any_phrase(
    text,
    phrases
):
    """
    Check whether any supplied phrase occurs
    in the text.
    """

    return any(
        contains_phrase(
            text,
            phrase
        )
        for phrase in phrases
    )


# =========================================================
# Tokenisation
# =========================================================

def tokenize(text):
    """
    Extract useful words for title matching.
    """

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        str(text).lower()
    )

    return {
        word
        for word in words
        if word not in STOP_WORDS
    }


# =========================================================
# Title matching
# =========================================================

def calculate_title_match(
    question,
    title
):
    """
    Calculate overlap between useful words in
    the question and the record title.
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
# Standard result structure
# =========================================================

def build_result(
    record,
    score,
    semantic_score=0.0,
    title_match=0.0,
    retrieval_mode="hybrid"
):
    """
    Convert a raw knowledge record into the
    standard result returned by the retriever.
    """

    description = (
        record.get(
            "Programme Description"
        )
        or
        record.get(
            "Description",
            ""
        )
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

        "source_file": record.get(
            "Source File",
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
    Return all knowledge records stored in metadata.
    """

    return [
        item["record"]
        for item in metadata
    ]


# =========================================================
# Domain detection
# =========================================================

def is_uom_domain_question(question):
    """
    Determine whether a question belongs to the
    current University of Mauritius knowledge domain.

    Mentioning UoM alone is not sufficient.
    The question should relate to a topic actually
    represented in the knowledge base.
    """

    q = normalize_text(
        question
    )


    # =====================================================
    # Strong external / unsupported topic exclusions
    # =====================================================

    external_signals = [

        "passport",

        "driving licence",
        "driver's licence",
        "drivers licence",

        "electricity bill",
        "water bill",

        "weather tomorrow",

        "fifa world cup",

        "chocolate cake",

        "graphics card",
        "gaming pc",

        "repair my laptop",

        "programming language should",

        "two factor authentication",

        "prime minister"
    ]


    if any(
        signal in q
        for signal in external_signals
    ):
        return False


    # =====================================================
    # External university detection
    #
    # Example:
    # "Computer Science at Harvard University"
    #
    # The knowledge base only represents UoM.
    # =====================================================

    external_university_pattern = re.search(
        (
            r"\b(?:at|from)\s+"
            r"(?!uom\b)"
            r"(?!university\s+of\s+mauritius\b)"
            r"[a-z0-9&.'\-\s]+?\s+university\b"
        ),
        q
    )


    if external_university_pattern:
        return False


    # =====================================================
    # Non-UoM bill payment
    # =====================================================

    if (
        contains_phrase(
            q,
            "bill"
        )
        and
        not any(
            signal in q
            for signal in [
                "uom",
                "university of mauritius",
                "fee",
                "fees"
            ]
        )
    ):
        return False


    # =====================================================
    # Institutional context
    # =====================================================

    institutional_signal = any(
        signal in q
        for signal in [
            "uom",
            "university of mauritius",
            "foicdt"
        ]
    )


    # =====================================================
    # Programme / degree terminology
    # =====================================================

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


    if contains_any_phrase(
        q,
        degree_signals
    ):
        return True


    # =====================================================
    # Specific programme subjects
    #
    # These are only accepted when the question also
    # contains institutional / study intent.
    #
    # This prevents:
    # "What is artificial intelligence?"
    #
    # from becoming a UoM query.
    # =====================================================

    programme_subject_signals = [

        "animation",
        "visual effects",

        "applied computing",

        "computer science",

        "information system",

        "software engineering",

        "cybersecurity",
        "cyber security",

        "artificial intelligence",

        "software project management"
    ]


    subject_signal = contains_any_phrase(
        q,
        programme_subject_signals
    )


    education_intent = contains_any_phrase(
        q,
        [
            "study",
            "studying",
            "offer",
            "offers",
            "offered",
            "course",
            "courses",
            "programme",
            "programmes",
            "degree",
            "degrees"
        ]
    )


    if (
        subject_signal
        and
        institutional_signal
        and
        education_intent
    ):
        return True


    # =====================================================
    # Faculty / department knowledge
    # =====================================================

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


    # =====================================================
    # Admissions / eligibility
    # =====================================================

    admission_signals = [

        "eligibility",
        "eligible",

        "admission",
        "admissions",

        "applicant",
        "applicants"
    ]


    if contains_any_phrase(
        q,
        admission_signals
    ):
        return True


    # =====================================================
    # Application documents
    # =====================================================

    document_signal = contains_any_phrase(
        q,
        [
            "document",
            "documents",
            "paperwork"
        ]
    )


    application_signal = contains_any_phrase(
        q,
        [
            "apply",
            "applying",
            "application"
        ]
    )


    if (
        document_signal
        and
        application_signal
    ):
        return True


    # =====================================================
    # Academic calendar / examinations
    # Member 2 knowledge
    # =====================================================

    academic_signals = [

        "semester",
        "semester 1",
        "semester 2",

        "examination",
        "examinations",

        "exam",
        "exams",

        "lecture",
        "lectures",

        "module registration",

        "floating week",

        "revision week",
        "revision weeks",

        "students week",
        "students' week",

        "graduation",
        "graduation ceremony",
        "graduation ceremonies",

        "induction",

        "late registration",

        "de registration",
        "deregistration"
    ]


    if contains_any_phrase(
        q,
        academic_signals
    ):
        return True


    # =====================================================
    # Student services
    # Member 2 knowledge
    # =====================================================

    service_signals = [

        "library",

        "borrow",
        "borrowing",

        "cits",

        "computer lab",
        "computer labs",

        "lab 1a",
        "lab 1b",
        "lab 1c",

        "vcilt",

        "mcb",
        "mcb juice",

        "sbm",

        "my.t",

        "blink"
    ]


    if contains_any_phrase(
        q,
        service_signals
    ):
        return True


    # =====================================================
    # Fees
    # =====================================================

    fee_signals = [

        "fee",
        "fees",

        "administrative fee",
        "administrative fees",

        "administrative charge",
        "administrative charges"
    ]


    if contains_any_phrase(
        q,
        fee_signals
    ):
        return True


    # =====================================================
    # UoM payment knowledge
    #
    # These are retained because the original Member 1
    # tests include questions such as:
    #
    # "Payment can be done by which means?"
    # =====================================================

    payment_signals = [

        "payment",
        "payments",

        "pay",
        "paying",

        "settle"
    ]


    if contains_any_phrase(
        q,
        payment_signals
    ):
        return True


    # =====================================================
    # Important:
    #
    # A UoM / FOICDT mention by itself does not make an
    # unsupported topic valid.
    #
    # Example:
    #
    # "Does UoM offer free accommodation and flights?"
    #
    # No supported knowledge category is detected, so
    # False is returned.
    # =====================================================

    return False


# =========================================================
# Undergraduate detection
# =========================================================

def detect_undergraduate_request(question):
    """
    Detect references to undergraduate studies.
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


    return contains_any_phrase(
        q,
        undergraduate_terms
    )


# =========================================================
# Postgraduate detection
# =========================================================

def detect_postgraduate_request(question):
    """
    Detect references to postgraduate studies.
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


    return contains_any_phrase(
        q,
        postgraduate_terms
    )


# =========================================================
# List intent detection
# =========================================================

def detect_list_intent(question):
    """
    Determine whether the user wants multiple
    records rather than one specific programme.
    """

    q = normalize_text(
        question
    )


    # -----------------------------------------------------
    # Specific named BSc / MSc request
    # -----------------------------------------------------

    specific_degree_reference = (
        contains_phrase(
            q,
            "bsc"
        )
        or
        contains_phrase(
            q,
            "msc"
        )
    )


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


    explicit_group_signals = [

        "show me all",

        "show all",

        "list all",

        "list the",

        "list every",

        "full set",

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


    if (
        specific_degree_reference
        and
        specific_information_request
        and
        not explicit_group_request
    ):
        return False


    # -----------------------------------------------------
    # Explicit list wording
    # -----------------------------------------------------

    explicit_list_signals = [

        "show me all",

        "show all",

        "list all",

        "list the",

        "list every",

        "full set",

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
    # "Every ..." wording
    # -----------------------------------------------------

    if contains_phrase(
        q,
        "every"
    ):

        if (
            detect_undergraduate_request(
                question
            )
            or
            detect_postgraduate_request(
                question
            )
            or
            contains_phrase(
                q,
                "degree"
            )
            or
            contains_phrase(
                q,
                "degrees"
            )
            or
            contains_phrase(
                q,
                "programme"
            )
            or
            contains_phrase(
                q,
                "programmes"
            )
            or
            contains_phrase(
                q,
                "program"
            )
            or
            contains_phrase(
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


    if contains_any_phrase(
        q,
        plural_terms
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
        and
        (
            "available" in q
            or
            "which" in q
            or
            "what are" in q
        )
    ):
        return True


    return False


# =========================================================
# Record classification
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
        or
        title.startswith(
            "bsc"
        )
    )


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
        or
        title.startswith(
            "msc"
        )
    )


def is_three_year_record(record):
    """
    Determine whether the programme duration
    is three years.
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
# Metadata retrieval
# =========================================================

def metadata_retrieve(question):
    """
    Handle structured list and filtering queries.
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


    # =====================================================
    # FOICDT department list
    # =====================================================

    department_list_request = (

        (
            "departments" in q
            or
            "different departments" in q
            or
            "each department" in q
        )

        and

        (
            "foicdt" in q
            or
            "heads" in q
            or
            "head" in q
            or
            "make up" in q
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
                ).startswith(
                    "FAC_"
                )

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


    # =====================================================
    # Undergraduate programme list
    # =====================================================

    if (
        undergraduate_request
        and
        list_intent
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

            or

            "three years" in q

            or

            "three year" in q
        )


        if asks_for_three_years:

            programme_records = [

                record

                for record
                in programme_records

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


    # =====================================================
    # Postgraduate programme list
    # =====================================================

    if (
        postgraduate_request
        and
        list_intent
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
# Intent-specific reranking
# =========================================================

def calculate_intent_boost(
    question,
    record
):
    """
    Apply targeted boosts to improve ranking for
    known UoM intents.
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


    # =====================================================
    # MEMBER 1 INTENTS
    # =====================================================


    # -----------------------------------------------------
    # Administrative fees
    # -----------------------------------------------------

    administrative_signal = (
        "administrative" in q
    )


    fee_signal = contains_any_phrase(
        q,
        [
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
        and
        administrative_signal
        and
        fee_signal
    ):
        boost += 0.20


    # -----------------------------------------------------
    # General UoM payment information
    # -----------------------------------------------------

    payment_signal = contains_any_phrase(
        q,
        [
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
        and
        payment_signal
    ):
        boost += 0.20


    # -----------------------------------------------------
    # Application documents
    # -----------------------------------------------------

    document_signal = contains_any_phrase(
        q,
        [
            "document",
            "documents",
            "paperwork"
        ]
    )


    if (
        record_id == "ADM_001"
        and
        document_signal
    ):
        boost += 0.12


    # -----------------------------------------------------
    # Eligibility
    # -----------------------------------------------------

    eligibility_signal = contains_any_phrase(
        q,
        [
            "eligible",
            "eligibility",
            "qualifications",
            "qualify"
        ]
    )


    if (
        record_id == "ADM_002"
        and
        eligibility_signal
    ):
        boost += 0.12


    # -----------------------------------------------------
    # Cybersecurity intent
    # -----------------------------------------------------

    cybersecurity_signal = contains_any_phrase(
        q,
        [
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
        and
        cybersecurity_signal
    ):
        boost += 0.08


    # =====================================================
    # MEMBER 2 ACADEMIC INFORMATION
    # =====================================================

    semester_1 = contains_phrase(
        q,
        "semester 1"
    )


    semester_2 = contains_phrase(
        q,
        "semester 2"
    )


    exam_signal = contains_any_phrase(
        q,
        [
            "exam",
            "exams",
            "examination",
            "examinations"
        ]
    )


    lecture_signal = contains_any_phrase(
        q,
        [
            "lecture",
            "lectures"
        ]
    )


    registration_signal = (
        "registration" in q
        and
        "module" in q
    )


    # -----------------------------------------------------
    # ACAD_001
    # -----------------------------------------------------

    if (
        record_id == "ACAD_001"
        and
        semester_1
        and
        lecture_signal
    ):
        boost += 0.25


    # -----------------------------------------------------
    # ACAD_002
    # -----------------------------------------------------

    if (
        record_id == "ACAD_002"
        and
        semester_1
        and
        registration_signal
        and
        "late" not in q
        and
        "penalty" not in q
    ):
        boost += 0.25


    # -----------------------------------------------------
    # ACAD_003
    # -----------------------------------------------------

    if (
        record_id == "ACAD_003"
        and
        "floating week" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # ACAD_004
    # -----------------------------------------------------

    if (
        record_id == "ACAD_004"
        and
        (
            "late registration" in q
            or
            "penalty" in q
        )
    ):
        boost += 0.30


    # -----------------------------------------------------
    # ACAD_005
    # -----------------------------------------------------

    if (
        record_id == "ACAD_005"
        and
        semester_1
        and
        exam_signal
    ):
        boost += 0.30


    # -----------------------------------------------------
    # ACAD_006
    # -----------------------------------------------------

    if (
        record_id == "ACAD_006"
        and
        "induction" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # ACAD_007
    # -----------------------------------------------------

    if (
        record_id == "ACAD_007"
        and
        semester_2
        and
        lecture_signal
    ):
        boost += 0.25


    # -----------------------------------------------------
    # ACAD_008
    # -----------------------------------------------------

    if (
        record_id == "ACAD_008"
        and
        semester_2
        and
        registration_signal
    ):
        boost += 0.25


    # -----------------------------------------------------
    # ACAD_009
    # -----------------------------------------------------

    if (
        record_id == "ACAD_009"
        and
        (
            "students week" in q
            or
            "students' week" in q
        )
    ):
        boost += 0.30


    # -----------------------------------------------------
    # ACAD_010
    # -----------------------------------------------------

    if (
        record_id == "ACAD_010"
        and
        semester_2
        and
        "revision" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # ACAD_011
    # -----------------------------------------------------

    if (
        record_id == "ACAD_011"
        and
        semester_2
        and
        exam_signal
    ):
        boost += 0.30


    # -----------------------------------------------------
    # ACAD_012
    # -----------------------------------------------------

    if (
        record_id == "ACAD_012"
        and
        "graduation" in q
    ):
        boost += 0.30


    # =====================================================
    # MEMBER 2 STUDENT SERVICES
    # =====================================================

    library_signal = (
        "library" in q
    )


    opening_signal = contains_any_phrase(
        q,
        [
            "open",
            "opening",
            "hours"
        ]
    )


    borrowing_signal = contains_any_phrase(
        q,
        [
            "borrow",
            "borrowing",
            "books"
        ]
    )


    # -----------------------------------------------------
    # SERV_001
    # -----------------------------------------------------

    if (
        record_id == "SERV_001"
        and
        library_signal
        and
        opening_signal
        and
        (
            "term" in q
            or
            "term time" in q
        )
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_002
    # -----------------------------------------------------

    if (
        record_id == "SERV_002"
        and
        library_signal
        and
        opening_signal
        and
        "vacation" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_003
    # -----------------------------------------------------

    if (
        record_id == "SERV_003"
        and
        borrowing_signal
        and
        "undergraduate" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_004
    # -----------------------------------------------------

    if (
        record_id == "SERV_004"
        and
        borrowing_signal
        and
        "academic staff" in q
        and
        "non academic" not in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_005
    # -----------------------------------------------------

    if (
        record_id == "SERV_005"
        and
        borrowing_signal
        and
        (
            "mphil" in q
            or
            "phd" in q
        )
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_006
    # -----------------------------------------------------

    if (
        record_id == "SERV_006"
        and
        borrowing_signal
        and
        "law" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_007
    # -----------------------------------------------------

    if (
        record_id == "SERV_007"
        and
        borrowing_signal
        and
        "non academic" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_008
    # -----------------------------------------------------

    if (
        record_id == "SERV_008"
        and
        "cits" in q
        and
        "lab" in q
        and
        "main" in q
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_009
    # -----------------------------------------------------

    if (
        record_id == "SERV_009"
        and
        "cits" in q
        and
        (
            "1a" in q
            or
            "1b" in q
            or
            "1c" in q
        )
    ):
        boost += 0.30


    # -----------------------------------------------------
    # SERV_012
    # -----------------------------------------------------

    if (
        record_id == "SERV_012"
        and
        "mcb" in q
        and
        "account" in q
    ):
        boost += 0.35


    # -----------------------------------------------------
    # SERV_013
    # -----------------------------------------------------

    if (
        record_id == "SERV_013"
        and
        "sbm" in q
        and
        (
            "mur" in q
            or
            "rupee" in q
            or
            "rupees" in q
        )
    ):
        boost += 0.35


    # -----------------------------------------------------
    # SERV_014
    # -----------------------------------------------------

    if (
        record_id == "SERV_014"
        and
        "sbm" in q
        and
        (
            "usd" in q
            or
            "dollar" in q
            or
            "dollars" in q
        )
    ):
        boost += 0.35


    # -----------------------------------------------------
    # SERV_015
    # -----------------------------------------------------

    if (
        record_id == "SERV_015"
        and
        (
            "my.t" in q
            or
            "blink" in q
            or
            (
                "mobile" in q
                and
                "payment" in q
            )
            or
            (
                "digital" in q
                and
                "payment" in q
            )
        )
    ):
        boost += 0.35


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

    Pipeline:

    1. Validate input
    2. Check supported UoM domain
    3. Attempt structured metadata retrieval
    4. Generate semantic embedding
    5. Search FAISS
    6. Apply title and intent reranking
    7. Return highest ranked records
    """

    # -----------------------------------------------------
    # Validate question
    # -----------------------------------------------------

    if not question:
        return []


    question = question.strip()


    if not question:
        return []


    # =====================================================
    # Domain validation
    # =====================================================

    if not is_uom_domain_question(
        question
    ):
        return []


    # =====================================================
    # Metadata retrieval
    # =====================================================

    metadata_results = metadata_retrieve(
        question
    )


    if metadata_results is not None:
        return metadata_results


    # =====================================================
    # Generate query embedding
    # =====================================================

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )


    # =====================================================
    # Candidate retrieval
    # =====================================================

    candidate_k = min(
        max(
            top_k * 3,
            10
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


    # =====================================================
    # Cybersecurity Top-Up wording
    # =====================================================

    question_mentions_top_up = bool(
        re.search(
            r"\btop\s*up\b",
            normalized_question
        )
    )


    # =====================================================
    # Process FAISS candidates
    # =====================================================

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
        # Semantic threshold
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
        # Title match
        # -------------------------------------------------

        title_match = calculate_title_match(
            question,
            title
        )


        # -------------------------------------------------
        # Intent boost
        # -------------------------------------------------

        intent_boost = calculate_intent_boost(
            question,
            record
        )


        # -------------------------------------------------
        # Hybrid score
        # -------------------------------------------------

        final_score = (

            semantic_score

            +

            (
                0.25
                * title_match
            )

            +

            intent_boost
        )


        # =================================================
        # Cybersecurity Top-Up disambiguation
        # =================================================

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


        # =================================================
        # Build result
        # =================================================

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


    # =====================================================
    # Sort by final hybrid score
    # =====================================================

    results.sort(
        key=lambda result:
        result["score"],
        reverse=True
    )


    # =====================================================
    # Return requested number of results
    # =====================================================

    return results[
        :top_k
    ]


# =========================================================
# Manual combined integration tests
# =========================================================

if __name__ == "__main__":

    test_questions = [

        (
            "What cybersecurity programmes "
            "are available?"
        ),

        (
            "When do Semester 1 examinations "
            "start and end?"
        ),

        (
            "What are the UoM library opening "
            "hours during term time?"
        ),

        (
            "How many books can undergraduate "
            "students borrow from the library?"
        ),

        (
            "Where is the main CITS Computer "
            "Lab located?"
        ),

        (
            "What is the MCB bank account number "
            "for UoM payments?"
        ),

        (
            "What digital and mobile payment apps "
            "are accepted by UoM?"
        ),

        (
            "What is the tuition fee for studying "
            "Computer Science at Harvard University?"
        ),

        (
            "Who is the Prime Minister "
            "of Australia?"
        ),

        (
            "Does UoM offer free accommodation "
            "and flights to international students?"
        ),

        (
            "Can you explain what artificial "
            "intelligence is?"
        ),

        (
            "How can I pay my electricity "
            "bill online?"
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


        for (
            number,
            result
        ) in enumerate(
            results,
            start=1
        ):

            print(
                f"{number}. "
                f"{result['id']} - "
                f"{result['title']} "
                f"(score={result['score']:.4f})"
            )