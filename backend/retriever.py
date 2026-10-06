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

    return [
        item["record"]
        for item in metadata
    ]


# =========================================================
# Domain detection
# =========================================================

def is_uom_domain_question(question):

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
    # =====================================================

    external_university_pattern = re.search(
        (
            r"\b(?:at|from)\s+"
            r"(?!uom\b)"
            r"(?!university\s+of\s+mauritius\b)"
            r"[a-z0-9&.'\s-]+?\s+university\b"
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
    # Unsupported accommodation + flights case
    # =====================================================

    if (
        (
            "accommodation" in q
            and
            contains_any_phrase(
                q,
                [
                    "flight",
                    "flights"
                ]
            )
        )
        or
        contains_any_phrase(
            q,
            [
                "free accommodation",
                "free flights"
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

    faculty_word_signal = contains_any_phrase(
        q,
        [
            "faculty",
            "faculties"
        ]
    )


    if (
        faculty_word_signal
        and
        (
            subject_signal
            or
            education_intent
            or
            institutional_signal
            or
            contains_any_phrase(
                q,
                [
                    "department",
                    "departments",

                    "faculty of agriculture",
                    "faculty of engineering",
                    "faculty of law",
                    "faculty of medicine",
                    "faculty of science",
                    "faculty of social sciences"
                ]
            )
        )
    ):

        return True


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
    # Payments
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
    # Member 3 contact information
    # =====================================================

    contact_signal = contains_any_phrase(
        q,
        [
            "contact",
            "telephone",
            "phone",
            "email",
            "address",
            "located"
        ]
    )


    if (
        contact_signal
        and
        (
            institutional_signal
            or
            contains_any_phrase(
                q,
                [
                    "admissions and student records office",
                    "asro",

                    "examinations office",
                    "examination office",

                    "international affairs office",

                    "student welfare office"
                ]
            )
        )
    ):

        return True


    # =====================================================
    # Member 3 regulations
    # =====================================================

    member3_regulation_signals = [

        "plagiarism",

        "academic integrity",

        "fabrication",
        "falsification",

        "turnitin",

        "resit",
        "resits",

        "retake",
        "retakes",

        "special retake",

        "final year project",
        "final year projects",

        "dissertation",
        "dissertations",

        "financial assistance",

        "student welfare",

        "scholarship",
        "scholarships",

        "student discipline",
        "disciplinary",

        "university rules",

        "attendance",

        "medical certificate",

        "unauthorised device",
        "unauthorized device",

        "regulation",
        "regulations",

        "withdraw from uom",
        "withdraw from the university"
    ]


    if contains_any_phrase(
        q,
        member3_regulation_signals
    ):

        return True


    # =====================================================
    # Member 3 assessment / credits / progression
    # =====================================================

    member3_academic_signals = [

        "continuous assessment",

        "gpa",
        "cpa",

        "grade n",

        "prerequisite",
        "pre requisite",
        "pre requirement",

        "credit system",
        "uom credits",

        "ncvts",

        "core module",
        "elective module",
        "audit module",

        "programme structure",
        "program structure",

        "repeat a year",
        "repeating a year",

        "termination of registration",

        "script review",
        "review of examination script",

        "academic dress"
    ]


    if contains_any_phrase(
        q,
        member3_academic_signals
    ):

        return True


    # =====================================================
    # Member 3 student support
    # =====================================================

    member3_support_signals = [

        "student union",
        "students' union",
        "students union",

        "first aid",

        "student counselling",
        "student counseling",

        "student welfare",

        "financial assistance",

        "student support",

        "international affairs office",

        "students with disabilities",
        "student disability",

        "student facilities"
    ]


    if contains_any_phrase(
        q,
        member3_support_signals
    ):

        return True


    # =====================================================
    # International student support topics
    # =====================================================

    if (
        (
            institutional_signal
            or
            contains_any_phrase(
                q,
                [
                    "international student",
                    "international students"
                ]
            )
        )
        and
        contains_any_phrase(
            q,
            [
                "accommodation",
                "visa",
                "residence permit",
                "medical insurance",

                "sports",
                "sports facilities",

                "counselling",
                "counseling"
            ]
        )
    ):

        return True


    # =====================================================
    # Academic AI usage
    #
    # Generic AI questions remain unsupported.
    # =====================================================

    ai_signal = contains_any_phrase(
        q,
        [
            "artificial intelligence",
            "ai"
        ]
    )


    ai_academic_use_signal = contains_any_phrase(
        q,
        [
            "university work",
            "academic work",

            "assignment",
            "assignments",

            "project",
            "projects",

            "dissertation",
            "dissertations",

            "cite ai",
            "citing ai",

            "acknowledge ai",

            "ai use",
            "use ai",

            "use artificial intelligence"
        ]
    )


    if (
        ai_signal
        and
        ai_academic_use_signal
    ):

        return True


    return False


# =========================================================
# Undergraduate detection
# =========================================================

def detect_undergraduate_request(question):

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

    q = normalize_text(
        question
    )


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
            contains_any_phrase(
                q,
                [
                    "degree",
                    "degrees",

                    "programme",
                    "programmes",

                    "program",
                    "programs"
                ]
            )
        ):

            return True


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
    # Existing Member 1 FOICDT department list
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
    # Helper for exact Member 3 metadata routes
    # =====================================================

    def result_for_id(record_id):

        record = next(
            (
                record
                for record in records
                if str(
                    record.get(
                        "ID",
                        ""
                    )
                ) == record_id
            ),
            None
        )


        if record is None:

            return None


        return [

            build_result(
                record=record,
                score=1.0,
                retrieval_mode="metadata"
            )
        ]


    # =====================================================
    # Member 3 contacts
    # =====================================================

    contact_signal = contains_any_phrase(
        q,
        [
            "contact",
            "telephone",
            "phone",
            "email",
            "address",
            "located"
        ]
    )


    if (
        contains_any_phrase(
            q,
            [
                "admissions and student records office",
                "asro"
            ]
        )
        and
        contact_signal
    ):

        return result_for_id(
            "M3_CONTACT_002"
        )


    if contains_any_phrase(
        q,
        [
            "examinations office",
            "examination office"
        ]
    ):

        return result_for_id(
            "M3_CONTACT_003"
        )


    if (
        "library" in q
        and
        contact_signal
    ):

        return result_for_id(
            "M3_CONTACT_004"
        )


    if (
        contains_any_phrase(
            q,
            [
                "international affairs office",
                "international student",
                "international students"
            ]
        )
        and
        contact_signal
    ):

        return result_for_id(
            "M3_CONTACT_005"
        )


    if (
        contact_signal
        and
        contains_any_phrase(
            q,
            [
                "uom",
                "university of mauritius"
            ]
        )
    ):

        return result_for_id(
            "M3_CONTACT_001"
        )


    # =====================================================
    # Member 3 faculties
    # =====================================================

    faculty_word_signal = contains_any_phrase(
        q,
        [
            "faculty",
            "faculties"
        ]
    )


    specific_faculty_phrases = [

        "faculty of agriculture",

        "faculty of engineering",

        "faculty of information",

        "faculty of law",

        "faculty of medicine",

        "faculty of science",

        "faculty of social sciences"
    ]


    faculty_list_request = (

        faculty_word_signal

        and

        not contains_any_phrase(
            q,
            specific_faculty_phrases
        )

        and

        contains_any_phrase(
            q,
            [
                "what faculties",
                "which faculties",
                "faculties available",
                "list faculties",
                "list the faculties",
                "how many faculties"
            ]
        )
    )


    if faculty_list_request:

        faculty_records = [

            record

            for record in records

            if str(
                record.get(
                    "ID",
                    ""
                )
            ).startswith(
                "M3_FAC_"
            )
        ]


        faculty_records.sort(
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
            in faculty_records
        ]


    cybersecurity_signal = contains_any_phrase(
        q,
        [
            "cybersecurity",
            "cyber security"
        ]
    )


    if (
        cybersecurity_signal
        and
        contains_any_phrase(
            q,
            [
                "faculty",
                "department"
            ]
        )
    ):

        return result_for_id(
            "M3_FAC_003"
        )


    faculty_department_map = {

        "faculty of agriculture":
            "M3_FAC_001",

        "faculty of engineering":
            "M3_FAC_002",

        "faculty of law & management":
            "M3_FAC_004",

        "faculty of law and management":
            "M3_FAC_004",

        "faculty of medicine and health sciences":
            "M3_FAC_005",

        "faculty of science":
            "M3_FAC_006",

        "faculty of social sciences & humanities":
            "M3_FAC_007",

        "faculty of social sciences and humanities":
            "M3_FAC_007"
    }


    if contains_phrase(
        q,
        "departments"
    ):

        for (
            faculty_phrase,
            record_id
        ) in faculty_department_map.items():

            if faculty_phrase in q:

                return result_for_id(
                    record_id
                )


    # =====================================================
    # Member 3 examination regulations
    # =====================================================

    examination_signal = contains_any_phrase(
        q,
        [
            "exam",
            "exams",

            "examination",
            "examinations"
        ]
    )


    illness_signal = contains_any_phrase(
        q,
        [
            "sick",
            "ill",
            "illness",

            "medical",
            "medical certificate",

            "miss",
            "missed",

            "absence",
            "absent"
        ]
    )


    if (
        examination_signal
        and
        illness_signal
    ):

        return result_for_id(
            "M3_EXAM_002"
        )


    if (
        examination_signal
        and
        contains_any_phrase(
            q,
            [
                "unauthorised device",
                "unauthorized device",

                "phone",
                "mobile phone",

                "device"
            ]
        )
    ):

        return result_for_id(
            "M3_EXAM_001"
        )


    # =====================================================
    # Member 3 academic integrity
    # =====================================================

    if contains_phrase(
        q,
        "plagiarism"
    ):

        return result_for_id(
            "M3_INT_001"
        )


    final_project_signal = contains_any_phrase(
        q,
        [
            "final year project",
            "final year projects",

            "dissertation",
            "dissertations"
        ]
    )


    ai_signal = contains_any_phrase(
        q,
        [
            "artificial intelligence",
            "ai"
        ]
    )


    if (
        final_project_signal
        and
        ai_signal
    ):

        return result_for_id(
            "M3_FYP_002"
        )


    if (
        ai_signal
        and
        contains_any_phrase(
            q,
            [
                "use",
                "using",

                "university work",
                "academic work",

                "assignment",
                "assignments",

                "cite",
                "citing",

                "acknowledge",

                "attributed"
            ]
        )
    ):

        return result_for_id(
            "M3_INT_002"
        )


    # =====================================================
    # Member 3 resits and retakes
    # =====================================================

    if (
        contains_any_phrase(
            q,
            [
                "resit",
                "resits"
            ]
        )
        and
        contains_any_phrase(
            q,
            [
                "retake",
                "retakes"
            ]
        )
    ):

        return result_for_id(
            "M3_ASSESS_003"
        )


    # =====================================================
    # Member 3 final year project / dissertation
    # =====================================================

    if (
        final_project_signal
        and
        contains_any_phrase(
            q,
            [
                "rule",
                "rules",

                "regulation",
                "regulations",

                "requirement",
                "requirements"
            ]
        )
    ):

        return result_for_id(
            "M3_FYP_001"
        )


    # =====================================================
    # Member 3 financial assistance
    # =====================================================

    if contains_any_phrase(
        q,
        [
            "financial assistance",
            "student financial assistance"
        ]
    ):

        return result_for_id(
            "M3_SUPPORT_004"
        )


    # =====================================================
    # Member 3 student conduct
    # =====================================================

    if contains_any_phrase(
        q,
        [
            "discipline",
            "disciplinary",

            "breaks university rules",
            "breach university rules",

            "student conduct"
        ]
    ):

        return result_for_id(
            "M3_CONDUCT_002"
        )


    # =====================================================
    # Existing Member 1 undergraduate programme lists
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
    # Existing Member 1 postgraduate programme lists
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


    if (
        record_id == "ACAD_001"
        and
        semester_1
        and
        lecture_signal
    ):

        boost += 0.25


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


    if (
        record_id == "ACAD_003"
        and
        "floating week" in q
    ):

        boost += 0.30


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


    if (
        record_id == "ACAD_005"
        and
        semester_1
        and
        exam_signal
    ):

        boost += 0.30


    if (
        record_id == "ACAD_006"
        and
        "induction" in q
    ):

        boost += 0.30


    if (
        record_id == "ACAD_007"
        and
        semester_2
        and
        lecture_signal
    ):

        boost += 0.25


    if (
        record_id == "ACAD_008"
        and
        semester_2
        and
        registration_signal
    ):

        boost += 0.25


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


    if (
        record_id == "ACAD_010"
        and
        semester_2
        and
        "revision" in q
    ):

        boost += 0.30


    if (
        record_id == "ACAD_011"
        and
        semester_2
        and
        exam_signal
    ):

        boost += 0.30


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


    if (
        record_id == "SERV_003"
        and
        borrowing_signal
        and
        "undergraduate" in q
    ):

        boost += 0.30


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


    if (
        record_id == "SERV_006"
        and
        borrowing_signal
        and
        "law" in q
    ):

        boost += 0.30


    if (
        record_id == "SERV_007"
        and
        borrowing_signal
        and
        "non academic" in q
    ):

        boost += 0.30


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


    if (
        record_id == "SERV_012"
        and
        "mcb" in q
        and
        "account" in q
    ):

        boost += 0.35


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


    # =====================================================
    # MEMBER 3 CONTACTS
    # =====================================================

    contact_signal = contains_any_phrase(
        q,
        [
            "contact",
            "telephone",
            "phone",
            "email",
            "address",
            "located"
        ]
    )


    if (
        record_id == "M3_CONTACT_001"
        and
        contact_signal
        and
        contains_any_phrase(
            q,
            [
                "uom",
                "university of mauritius"
            ]
        )
    ):

        boost += 0.40


    if (
        record_id == "M3_CONTACT_002"
        and
        contains_any_phrase(
            q,
            [
                "admissions and student records office",
                "asro"
            ]
        )
    ):

        boost += 0.45


    if (
        record_id == "M3_CONTACT_003"
        and
        contains_any_phrase(
            q,
            [
                "examinations office",
                "examination office"
            ]
        )
    ):

        boost += 0.45


    if (
        record_id == "M3_CONTACT_004"
        and
        "library" in q
        and
        contact_signal
    ):

        boost += 0.45


    # =====================================================
    # MEMBER 3 FACULTIES
    # =====================================================

    if (
        record_id == "M3_FAC_003"
        and
        cybersecurity_signal
        and
        contains_any_phrase(
            q,
            [
                "faculty",
                "department"
            ]
        )
    ):

        boost += 0.50


    # =====================================================
    # MEMBER 3 EXAMINATIONS
    # =====================================================

    if (
        record_id == "M3_EXAM_002"
        and
        contains_any_phrase(
            q,
            [
                "sick",
                "ill",
                "illness",

                "medical certificate",

                "absence",
                "absent",

                "miss",
                "missed"
            ]
        )
        and
        exam_signal
    ):

        boost += 0.45


    if (
        record_id == "M3_EXAM_001"
        and
        exam_signal
        and
        contains_any_phrase(
            q,
            [
                "unauthorised device",
                "unauthorized device",
                "device",
                "phone"
            ]
        )
    ):

        boost += 0.45


    # =====================================================
    # MEMBER 3 ACADEMIC INTEGRITY
    # =====================================================

    if (
        record_id == "M3_INT_001"
        and
        "plagiarism" in q
    ):

        boost += 0.50


    if (
        record_id == "M3_INT_002"
        and
        contains_any_phrase(
            q,
            [
                "artificial intelligence",
                "ai"
            ]
        )
        and
        contains_any_phrase(
            q,
            [
                "use",
                "using",

                "university work",
                "academic work",

                "assignment",
                "assignments"
            ]
        )
    ):

        boost += 0.50


    # =====================================================
    # MEMBER 3 ASSESSMENT
    # =====================================================

    if (
        record_id == "M3_ASSESS_003"
        and
        contains_any_phrase(
            q,
            [
                "resit",
                "resits",

                "retake",
                "retakes"
            ]
        )
    ):

        boost += 0.50


    # =====================================================
    # MEMBER 3 FINAL YEAR PROJECT
    # =====================================================

    if (
        record_id == "M3_FYP_001"
        and
        contains_any_phrase(
            q,
            [
                "final year project",
                "final year projects",

                "dissertation",
                "dissertations"
            ]
        )
    ):

        boost += 0.45


    # =====================================================
    # MEMBER 3 FINANCIAL ASSISTANCE
    # =====================================================

    if (
        record_id == "M3_SUPPORT_004"
        and
        contains_any_phrase(
            q,
            [
                "financial assistance",
                "student welfare"
            ]
        )
    ):

        boost += 0.50


    # =====================================================
    # MEMBER 3 STUDENT CONDUCT
    # =====================================================

    if (
        record_id == "M3_CONDUCT_002"
        and
        contains_any_phrase(
            q,
            [
                "discipline",
                "disciplinary",

                "university rules",

                "student conduct"
            ]
        )
    ):

        boost += 0.50


    return boost


# =========================================================
# Main retrieval function
# =========================================================

def retrieve(
    question,
    top_k=3,
    min_semantic_score=DEFAULT_MIN_SEMANTIC_SCORE
):

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
        ),

        (
            "How can I contact the "
            "University of Mauritius?"
        ),

        (
            "What counts as plagiarism "
            "at UoM?"
        ),

        (
            "Can I use Artificial Intelligence "
            "in my university work?"
        ),

        (
            "What should I do if I miss an "
            "examination because I am sick?"
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