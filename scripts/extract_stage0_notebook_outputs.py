"""Extract executed Stage 0 outputs from a Colab notebook backup.

This is a read-only extractor for user-provided `.ipynb` files. It does not
execute notebook code. It converts visible output tables and streams into local
CSV/TXT/JSON artifacts for review.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from bs4 import BeautifulSoup
import nbformat
import pandas as pd


TARGET_CELLS = {
    13: "stage0a1",
    15: "stage0a2",
    17: "stage0b",
    20: "drive_upload",
}


def parse_html_table(html: str) -> pd.DataFrame:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError("No HTML table found")
    rows = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(strip=True) for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append(cells)
    if len(rows) < 2:
        return pd.DataFrame()
    header = rows[0]
    if header and header[0] == "":
        header = header[1:]
        data_rows = [row[1:] if len(row) == len(header) + 1 else row for row in rows[1:]]
    else:
        data_rows = rows[1:]
    width = len(header)
    normalized = []
    for row in data_rows:
        if len(row) < width:
            row = row + [""] * (width - len(row))
        normalized.append(row[:width])
    frame = pd.DataFrame(normalized, columns=header)
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="ignore")
    return frame


def extract_json_blocks(text: str) -> list[dict]:
    blocks: list[dict] = []
    starts = [match.start() for match in re.finditer(r"\{\s*\"", text)]
    for start in starts:
        decoder = json.JSONDecoder()
        try:
            payload, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            blocks.append(payload)
    return blocks


def output_text(output) -> str:
    if output.get("output_type") == "stream":
        return output.get("text", "")
    data = output.get("data", {})
    return data.get("text/plain", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(notebook_path, as_version=4)
    metadata = {
        "source_notebook": str(notebook_path),
        "cell_count": len(nb.cells),
        "code_cell_count": sum(cell.cell_type == "code" for cell in nb.cells),
        "executed_code_cell_count": sum(
            cell.cell_type == "code" and cell.execution_count is not None for cell in nb.cells
        ),
        "output_count": sum(len(cell.get("outputs", [])) for cell in nb.cells),
        "error_outputs": [],
        "kernelspec": nb.metadata.get("kernelspec", {}),
    }

    all_json_blocks: list[dict] = []
    extracted_tables: dict[str, str] = {}
    streams: dict[str, str] = {}

    for index, cell in enumerate(nb.cells, start=1):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                metadata["error_outputs"].append(
                    {
                        "cell": index,
                        "ename": output.get("ename"),
                        "evalue": output.get("evalue"),
                    }
                )

        label = TARGET_CELLS.get(index)
        if label is None:
            continue

        combined_text = []
        table_counter = 0
        for output in cell.get("outputs", []):
            text = output_text(output)
            if text:
                combined_text.append(text)
                all_json_blocks.extend(extract_json_blocks(text))

            html = output.get("data", {}).get("text/html")
            if html:
                table_counter += 1
                frame = parse_html_table(html)
                table_path = out_dir / f"{label}_table{table_counter}.csv"
                frame.to_csv(table_path, index=False)
                extracted_tables[f"{label}_table{table_counter}"] = str(table_path)

        if combined_text:
            text_path = out_dir / f"{label}_stream.txt"
            text_path.write_text("\n".join(combined_text), encoding="utf-8")
            streams[label] = str(text_path)

    # Most stage decision JSON blocks are printed in order: 0A1 then 0A2.
    decision_blocks = [
        block for block in all_json_blocks if "stage0_result" in block or "candidate_count" in block
    ]
    decisions_path = out_dir / "stage0_decision_blocks.json"
    decisions_path.write_text(json.dumps(decision_blocks, indent=2), encoding="utf-8")

    summary = {
        "metadata": metadata,
        "tables": extracted_tables,
        "streams": streams,
        "decision_blocks": str(decisions_path),
    }
    (out_dir / "extraction_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
