"""Validate the published notebook without running training, downloads or inference."""

import ast
from pathlib import Path
import re

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "bert_imdb_sentiment_analysis.ipynb"


def main():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    code_count = 0
    markdown_count = 0
    credentials = re.compile(
        r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{20,})\b"
    )
    literal_credential = re.compile(
        r"(?:api_key|access_token|password)\s*=\s*['\"][A-Za-z0-9_\-]{16,}['\"]", re.I
    )
    for index, cell in enumerate(notebook.cells):
        source = cell.source
        if credentials.search(source) or literal_credential.search(source):
            raise ValueError(f"Cell {index + 1}: possible hardcoded credential")
        if cell.cell_type == "markdown":
            markdown_count += 1
            if len(re.findall(r"^```", source, re.M)) % 2:
                raise ValueError(f"Cell {index + 1}: unbalanced Markdown code fence")
        elif cell.cell_type == "code":
            code_count += 1
            # The supplied notebook has a bare IPython `pip install emoji` cell.
            # Recognize this form for syntax validation; never execute it here.
            cleaned = "\n".join(
                "# IPython command" if re.match(r"^\s*(?:%pip\b|!pip\b|pip\s+install\b)", line) else line
                for line in source.splitlines()
            )
            ast.parse(cleaned, filename=f"{NOTEBOOK.name}:cell-{index + 1}")
    if not code_count or not markdown_count:
        raise ValueError("Expected both code and Markdown cells")
    print(f"Notebook structure valid: {code_count} code cells, {markdown_count} Markdown cells.")
    print("No notebook cells were executed; historical metrics were not re-evaluated.")


if __name__ == "__main__":
    main()
