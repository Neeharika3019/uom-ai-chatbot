import json
import os

import faiss
from sentence_transformers import SentenceTransformer


PROGRAMMES_FILE = "data/processed/programmes.json"
ADMISSIONS_FILE = "data/processed/admissions_faculty.json"

VECTOR_DB_DIR = "vector_db"

INDEX_FILE = os.path.join(VECTOR_DB_DIR, "uom_index.faiss")
METADATA_FILE = os.path.join(VECTOR_DB_DIR, "metadata.json")

MODEL_NAME = "all-MiniLM-L6-v2"


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def create_search_text(record):
    parts = []

    if record.get("Title"):
        parts.append(f"Title: {record['Title']}")

    if record.get("Category"):
        parts.append(f"Category: {record['Category']}")

    if record.get("Subcategory"):
        parts.append(f"Subcategory: {record['Subcategory']}")

    if record.get("Faculty"):
        parts.append(f"Faculty: {record['Faculty']}")

    if record.get("Duration"):
        parts.append(f"Duration: {record['Duration']}")

    if record.get("Academic Year"):
        parts.append(f"Academic Year: {record['Academic Year']}")

    if record.get("Programme Description"):
        parts.append(
            f"Programme Description: {record['Programme Description']}"
        )

    if record.get("Description"):
        parts.append(
            f"Description: {record['Description']}"
        )

    return ". ".join(parts)


def main():

    print("Loading processed datasets...")

    programmes = load_json(PROGRAMMES_FILE)
    admissions = load_json(ADMISSIONS_FILE)

    records = programmes + admissions

    print(f"Total records loaded: {len(records)}")

    search_texts = []

    for record in records:
        search_text = create_search_text(record)
        search_texts.append(search_text)

    print("\nExample searchable text:\n")
    print(search_texts[0])

    print("\nLoading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    print("Generating embeddings...")

    embeddings = model.encode(
        search_texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    dimension = embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    # Normalized embeddings + inner product = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    print(f"Number of vectors stored in FAISS: {index.ntotal}")

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    faiss.write_index(index, INDEX_FILE)

    metadata = []

    for record, search_text in zip(records, search_texts):
        metadata.append(
            {
                "record": record,
                "search_text": search_text
            }
        )

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

    print("\nVector database created successfully.")
    print(f"FAISS index saved to: {INDEX_FILE}")
    print(f"Metadata saved to: {METADATA_FILE}")


if __name__ == "__main__":
    main()