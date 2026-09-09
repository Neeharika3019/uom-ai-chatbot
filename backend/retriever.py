import json

import faiss
from sentence_transformers import SentenceTransformer


INDEX_FILE = "vector_db/uom_index.faiss"
METADATA_FILE = "vector_db/metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"


# Load the same embedding model used to build the vector database
print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)


# Load FAISS vector index
print("Loading FAISS index...")
index = faiss.read_index(INDEX_FILE)


# Load metadata associated with the FAISS vectors
with open(METADATA_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)


def retrieve(question, top_k=3):
    """
    Retrieve the most relevant UoM records for a user question.
    """

    # Convert the user's question into an embedding
    question_embedding = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Ensure top_k is not greater than the number of records
    top_k = min(top_k, index.ntotal)

    # Search FAISS
    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, index_position in zip(scores[0], indices[0]):

        if index_position == -1:
            continue

        item = metadata[index_position]
        record = item["record"]

        # Some records use Programme Description,
        # while admissions/faculty records use Description.
        description = (
            record.get("Programme Description")
            or record.get("Description", "")
        )

        result = {
            "id": record.get("ID", ""),
            "title": record.get("Title", ""),
            "category": record.get("Category", ""),
            "subcategory": record.get("Subcategory", ""),
            "faculty": record.get("Faculty", ""),
            "duration": record.get("Duration", ""),
            "applicant_type": record.get("Applicant Type", ""),
            "description": description,
            "contact_email": record.get("Contact Email", ""),
            "source_url": record.get("Source URL", ""),
            "score": float(score)
        }

        results.append(result)

    return results


if __name__ == "__main__":

    test_questions = [
        "What cybersecurity programmes are available?",
        "How long is the BSc Cyber Security programme?",
        "What documents are required when applying to UoM?",
        "Who is the Dean of FOICDT?"
    ]

    for question in test_questions:

        print("\n" + "=" * 70)
        print("QUESTION:")
        print(question)

        results = retrieve(question, top_k=3)

        print("\nTOP 3 RESULTS:")

        for number, result in enumerate(results, start=1):

            print(f"\nResult {number}")
            print(f"ID: {result['id']}")
            print(f"Title: {result['title']}")
            print(f"Category: {result['category']}")
            print(f"Subcategory: {result['subcategory']}")
            print(f"Score: {result['score']:.4f}")

            if result["duration"]:
                print(f"Duration: {result['duration']}")

            if result["contact_email"]:
                print(f"Contact Email: {result['contact_email']}")

            print(f"Source URL: {result['source_url']}")