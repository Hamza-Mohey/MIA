"""Download the small semantic-retrieval model once for offline use."""

from sentence_transformers import SentenceTransformer

from modules.config import HF_CACHE_DIR, RETRIEVAL_MODEL_NAME


def main() -> None:
    SentenceTransformer(RETRIEVAL_MODEL_NAME, cache_folder=str(HF_CACHE_DIR), device="cpu")
    print(f"Retrieval model cached in {HF_CACHE_DIR}")


if __name__ == "__main__":
    main()
