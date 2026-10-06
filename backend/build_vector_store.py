import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# =========================================================
# Project paths
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

VECTOR_DB_DIR = (
    PROJECT_ROOT
    / "vector_db"
)


PROGRAMMES_FILE = (
    PROCESSED_DIR
    / "programmes.json"
)

ADMISSIONS_FILE = (
    PROCESSED_DIR
    / "admissions_faculty.json"
)

MEMBER2_FILE = (
    PROCESSED_DIR
    / "member2_knowledge.json"
)

MEMBER3_FILE = (
    PROCESSED_DIR
    / "member3_knowledge.json"
)


INDEX_FILE = (
    VECTOR_DB_DIR
    / "uom_index.faiss"
)

METADATA_FILE = (
    VECTOR_DB_DIR
    / "metadata.json"
)


# =========================================================
# Embedding model
# =========================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# =========================================================
# Load JSON records
# =========================================================

def load_json_records(filepath):
    """
    Load structured records from a JSON file.
    """

    if not filepath.exists():

        raise FileNotFoundError(
            f"File not found: {filepath}"
        )


    with open(
        filepath,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)


    if not isinstance(
        data,
        list
    ):

        raise ValueError(
            f"Expected a list of records in {filepath}"
        )


    return data


# =========================================================
# Create searchable text
# =========================================================

def create_search_text(record):
    """
    Convert a structured record into a single text
    representation for embedding and semantic search.
    """

    searchable_fields = [
        (
            "Title",
            record.get(
                "Title",
                ""
            )
        ),
        (
            "Category",
            record.get(
                "Category",
                ""
            )
        ),
        (
            "Subcategory",
            record.get(
                "Subcategory",
                ""
            )
        ),
        (
            "Faculty",
            record.get(
                "Faculty",
                ""
            )
        ),
        (
            "Duration",
            record.get(
                "Duration",
                ""
            )
        ),
        (
            "Academic Year",
            record.get(
                "Academic Year",
                ""
            )
        ),
        (
            "Programme Description",
            record.get(
                "Programme Description",
                ""
            )
        ),
        (
            "Description",
            record.get(
                "Description",
                ""
            )
        ),
        (
            "Applicant Type",
            record.get(
                "Applicant Type",
                ""
            )
        ),
        (
            "Contact Email",
            record.get(
                "Contact Email",
                ""
            )
        ),
        (
            "Source File",
            record.get(
                "Source File",
                ""
            )
        )
    ]


    text_parts = []


    for (
        label,
        value
    ) in searchable_fields:

        value = str(
            value
        ).strip()


        if value:

            text_parts.append(
                f"{label}: {value}"
            )


    return " | ".join(
        text_parts
    )


# =========================================================
# Main vector-store build
# =========================================================

def main():

    print(
        "Loading processed datasets..."
    )


    # -----------------------------------------------------
    # Load Member 1 datasets
    # -----------------------------------------------------

    programme_records = (
        load_json_records(
            PROGRAMMES_FILE
        )
    )

    admission_records = (
        load_json_records(
            ADMISSIONS_FILE
        )
    )


    # -----------------------------------------------------
    # Load Member 2 dataset
    # -----------------------------------------------------

    member2_records = (
        load_json_records(
            MEMBER2_FILE
        )
    )


    # -----------------------------------------------------
    # Load Member 3 dataset
    # -----------------------------------------------------

    member3_records = (
        load_json_records(
            MEMBER3_FILE
        )
    )


    # -----------------------------------------------------
    # Combine all knowledge records
    # -----------------------------------------------------

    all_records = (
        programme_records
        + admission_records
        + member2_records
        + member3_records
    )


    print(
        f"Programme records: "
        f"{len(programme_records)}"
    )

    print(
        f"Admissions/faculty records: "
        f"{len(admission_records)}"
    )

    print(
        f"Member 2 records: "
        f"{len(member2_records)}"
    )

    print(
        f"Member 3 records: "
        f"{len(member3_records)}"
    )

    print(
        f"Total records: "
        f"{len(all_records)}"
    )


    # -----------------------------------------------------
    # Check duplicate record IDs
    # -----------------------------------------------------

    record_ids = [
        str(
            record.get(
                "ID",
                ""
            )
        ).strip()
        for record in all_records
    ]


    if any(
        not record_id
        for record_id in record_ids
    ):

        raise ValueError(
            "One or more records do not contain an ID."
        )


    if (
        len(record_ids)
        != len(set(record_ids))
    ):

        raise ValueError(
            "Duplicate record IDs detected."
        )


    # -----------------------------------------------------
    # Create search text
    # -----------------------------------------------------

    metadata = []

    search_texts = []


    for record in all_records:

        search_text = (
            create_search_text(
                record
            )
        )


        if not search_text:

            raise ValueError(
                "A record produced empty search text: "
                f"{record.get('ID', 'UNKNOWN')}"
            )


        metadata.append(
            {
                "record": record,
                "search_text": search_text
            }
        )


        search_texts.append(
            search_text
        )


    # -----------------------------------------------------
    # Load embedding model
    # -----------------------------------------------------

    print(
        "Loading embedding model..."
    )


    model = SentenceTransformer(
        MODEL_NAME
    )


    # -----------------------------------------------------
    # Generate embeddings
    # -----------------------------------------------------

    print(
        "Generating embeddings..."
    )


    embeddings = model.encode(
        search_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    )


    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )


    print(
        f"Embedding dimension: "
        f"{embeddings.shape[1]}"
    )


    # -----------------------------------------------------
    # Create FAISS index
    # -----------------------------------------------------

    print(
        "Building FAISS index..."
    )


    dimension = embeddings.shape[1]


    index = faiss.IndexFlatIP(
        dimension
    )


    index.add(
        embeddings
    )


    print(
        f"Vectors stored: "
        f"{index.ntotal}"
    )


    # -----------------------------------------------------
    # Ensure output directory exists
    # -----------------------------------------------------

    VECTOR_DB_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # -----------------------------------------------------
    # Save FAISS index
    # -----------------------------------------------------

    faiss.write_index(
        index,
        str(
            INDEX_FILE
        )
    )


    # -----------------------------------------------------
    # Save metadata
    # -----------------------------------------------------

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )


    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print(
        "\nVector store created successfully."
    )

    print(
        f"FAISS index: "
        f"{INDEX_FILE}"
    )

    print(
        f"Metadata: "
        f"{METADATA_FILE}"
    )

    print(
        f"Total indexed records: "
        f"{index.ntotal}"
    )


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    main()