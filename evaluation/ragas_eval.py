import argparse
import json
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from src.search import RAGSearch


def _load_dataset(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError("Dataset file must contain a JSON array.")

    cleaned = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {idx} is not an object.")
        question = str(row.get("question", "")).strip()
        ground_truth = str(row.get("ground_truth", "")).strip()
        if not question or not ground_truth:
            raise ValueError(f"Row {idx} requires non-empty 'question' and 'ground_truth'.")
        cleaned.append({"question": question, "ground_truth": ground_truth})
    return cleaned


def _build_eval_rows(rag: RAGSearch, rows: list[dict], top_k: int) -> list[dict]:
    output = []
    for row in rows:
        result = rag.answer_with_sources(query=row["question"], top_k=top_k)
        contexts = [str(source.get("text", "") or "") for source in result.get("sources", [])]
        output.append(
            {
                "question": row["question"],
                "answer": str(result.get("answer", "") or ""),
                "contexts": contexts,
                "ground_truth": row["ground_truth"],
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation for the local RAG pipeline.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to a JSON array test set")
    parser.add_argument("--top-k", type=int, default=3, help="Retriever top-k")
    parser.add_argument("--persist-dir", type=str, default="faiss_store", help="Vector store directory")
    parser.add_argument("--llm-model", type=str, default=os.getenv("GOOGLE_LLM_MODEL", "gemma-3-12b-it"))
    args = parser.parse_args()

    load_dotenv()
    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing. Add it to .env before running evaluation.")

    questions = _load_dataset(dataset_path)
    rag = RAGSearch(persist_dir=args.persist_dir)
    eval_rows = _build_eval_rows(rag, questions, top_k=args.top_k)
    eval_dataset = Dataset.from_list(eval_rows)

    eval_llm = ChatGoogleGenerativeAI(google_api_key=api_key, model=args.llm_model)
    eval_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
    )

    print("\nRAGAS evaluation complete")
    print(result)


if __name__ == "__main__":
    main()
