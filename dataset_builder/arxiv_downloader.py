"""
ArxivDownloader: Downloads full-text PDFs from arXiv.org given a paper_id.

Rate-limits requests to 3s delay between downloads to comply with arXiv ToS.
Caches downloads locally to avoid re-downloading on repeated runs.
"""

import os
import re
import time
import requests


class ArxivDownloader:
    BASE_URL = "https://arxiv.org/pdf/{paper_id}.pdf"

    def __init__(self, save_dir: str = "./data/arxiv_pdfs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def normalize_id(paper_id: str) -> str:
        """
        Normalize arXiv paper IDs from Kaggle dataset format.

        Kaggle stores some IDs without a leading zero:
          '704.0047'  → '0704.0047'   (April 2007)
          '704.005'   → '0704.005'

        Standard new-format IDs are left unchanged:
          '2301.12345' → '2301.12345'

        Old category-prefixed IDs are left unchanged:
          'cs/0704.0047' → 'cs/0704.0047'
        """
        # Old format with category prefix — leave as-is
        if "/" in paper_id:
            return paper_id

        # Missing leading zero: "704.XXXX" → "0704.XXXX"
        if re.match(r"^\d{3}\.\d+$", paper_id):
            return "0" + paper_id

        return paper_id

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
        normalized_id = self.normalize_id(paper_id)
        safe_id = normalized_id.replace("/", "_")
        pdf_path = os.path.join(self.save_dir, f"{safe_id}.pdf")

        # Cache hit: skip download
        if os.path.exists(pdf_path):
            print(f"  [cache] {normalized_id}")
            return pdf_path

        url = self.BASE_URL.format(paper_id=normalized_id)

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
                    print(f"  [ok] {normalized_id} → {pdf_path}")
                    return pdf_path
                else:
                    print(f"  [!] {normalized_id}: HTTP {response.status_code} (attempt {attempt})")
            except Exception as e:
                print(f"  [!] {normalized_id}: {e} (attempt {attempt})")

        print(f"  [fail] Could not download {normalized_id} after {max_retries} attempts.")
        return None
