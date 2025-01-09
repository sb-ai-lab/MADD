from pathlib import Path

from langchain_core.documents import Document


def prepare_documents(path: Path):
    """
    iterates all .txt files in directory and returns documents
    """

    docs = []

    for p in path.glob("**/*.txt"):
        with open(str(p.resolve()), "r", encoding="utf-8") as f:
            docs.append(Document(page_content=f.read()))

    return docs