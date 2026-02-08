#!/usr/bin/env python3
import re
import json
from pathlib import Path

from pymilvus import MilvusClient
# BGE-M3 model can embed texts as dense and sparse vectors.
# It is included in the optional `model` module in pymilvus, to install it,
# simply run "pip install pymilvus[model]".
from pymilvus.model.hybrid import BGEM3EmbeddingFunction


ef = BGEM3EmbeddingFunction(use_fp16=False, device="mps")
from langchain_text_splitters import RecursiveCharacterTextSplitter

MD_LINK = re.compile(r"\[([^\]\n]*)\]\(([^)\s]+)\)")

def extract_md_links(text: str) -> tuple[str, list[str]]:
    links: list[str] = []
    def repl(match):
        label, url = match.group(1), match.group(2)
        links.append(f"[{label}]({url})")
        return f"[{label}]"

    new_text = MD_LINK.sub(repl, text)
    return new_text, links

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50
)

client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus"
)

def normalize_whitespace(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\s+", " ", text).strip()

def walk_and_process_json(
    root: str | Path,
    *,
    process: callable = None,
    skip_dirs: set | None = None,
):
    root = Path(root)
    skip_dirs = skip_dirs or set()

    def default_process(path: Path, data: dict):
        print(path.relative_to(root), "— keys:", list(data.keys())[:5])

    process = process or default_process

    for path in root.rglob("*.json"):
        if any(part in skip_dirs for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Skip {path}: {e}")
            continue
        process(path, data)


def inference(path: Path, data: dict):
    print(f'handle {path}')
    title = data.get("title")
    canonical = data.get("canonical")
    if not canonical:
        canonical = data.get("alternate")

    if canonical:
        canonical = canonical.replace('ru/', 'ru/docs/').replace('.html', '')

    text = data.get("text")
    if text:
        text = normalize_whitespace(text)

    new_text, links = extract_md_links(text)
    texts = [title, new_text]
    docs = text_splitter.create_documents(texts)
    docs = [doc.page_content for doc in docs]
    docs_embeddings = ef(docs)
    dense = docs_embeddings["dense"]
    sparse = docs_embeddings["sparse"]
    # sparse от BGE-M3 — одна csr-матрица (n_docs, dim). PyMilvus принимает только 1-row scipy sparse или dict/list пар.
    collection = []
    for i, embedding in enumerate(dense):
        # одна строка как 1-row (shape (1, dim)); csr_array не имеет getrow, используем срез
        sparse_row = sparse[i : i + 1]
        collection.append(
            {
                "pk": f"{canonical}_{i}",
                "url": canonical,
                "text": docs[i],
                "metadata": {
                    "title": title,
                    "links": links,
                    "chunk": i
                },
                "sparse_vector": sparse_row,
                "dense_vector": embedding.tolist() if hasattr(embedding, "tolist") else embedding
            }
        )
    try:
        client.insert(
            collection_name="ru_docs",
            data=collection
        )
        print(f'success index {path}')
    except Exception as _e:
        print(f"error: {_e}, parse page: {canonical}")

if __name__ == "__main__":
    import sys

    folder = sys.argv[1] if len(sys.argv) > 1 else "./tmp"
    walk_and_process_json(folder, process=inference, skip_dirs={"node_modules", ".git"})
