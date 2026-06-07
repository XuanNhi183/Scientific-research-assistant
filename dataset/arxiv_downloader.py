"""
ArxivDownloader: Downloads full-text PDFs from arXiv.org given a paper_id.

Rate-limits requests to 3s delay between downloads to comply with arXiv ToS.
Caches downloads locally to avoid re-downloading on repeated runs.
"""

import os
import time
import requests


class ArxivDownloader:
    BASE_URL = "https://arxiv.org/pdf/{paper_id}.pdf"

    def __init__(self, save_dir: str = "./data/arxiv_pdfs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def download(
        self,
        paper_id: str,
        delay: float = 3.0,
        max_retries: int = 3,
    ) -> str | None:
        """
        Download a PDF from arXiv and return the local file path.
        Returns None if all retries fail.
        """
        # Normalize paper_id (e.g. "2301.12345v1" → "2301.12345v1")
        safe_id = paper_id.replace("/", "_")
        pdf_path = os.path.join(self.save_dir, f"{safe_id}.pdf")

        # Cache hit: skip download
        if os.path.exists(pdf_path):
            print(f"  [cache] {paper_id}")
            return pdf_path

        url = self.BASE_URL.format(paper_id=paper_id)

        for attempt in range(1, max_retries + 1):
            try:
                time.sleep(delay)
                response = requests.get(
                    url,
                    timeout=30,
                    headers={"User-Agent": "DatasetBuilder/1.0"},
                )
                if response.status_code == 200 and response.content[:4] == b"%PDF":
                    with open(pdf_path, "wb") as f:
                        f.write(response.content)
                    print(f"  [ok] {paper_id} → {pdf_path}")
                    return pdf_path
                else:
                    print(f"  [!] {paper_id}: HTTP {response.status_code} (attempt {attempt})")
            except Exception as e:
                print(f"  [!] {paper_id}: {e} (attempt {attempt})")

        print(f"  [fail] Could not download {paper_id} after {max_retries} attempts.")
        return None
