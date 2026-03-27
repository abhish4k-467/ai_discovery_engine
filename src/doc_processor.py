import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from typing import Any


REQUEST_TIMEOUT_SECONDS = 30
DEFAULT_HEADERS = {
    "User-Agent": "AI-Discovery-Engine/1.0",
}


class DocumentProcessor:
    def process_pdf(self, file_path: str) -> list[dict[str, Any]]:
        """Extracts text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n".join(filter(None, pages)).strip()
            if not text:
                return []
            return self._chunk_text(text, source=file_path)
        except Exception as e:
            print(f"Error processing PDF {file_path}: {e}")
            return []

    def process_url(self, url: str) -> list[dict[str, Any]]:
        """Scrapes text from a webpage."""
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS, headers=DEFAULT_HEADERS)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")
            # Extract text from p, h1, h2, h3 tags for content
            content = []
            for tag in soup.find_all(["p", "h1", "h2", "h3", "li"]):
                content.append(tag.get_text(strip=True))
            text = "\n".join(filter(None, content)).strip()
            if not text:
                return []
            return self._chunk_text(text, source=url)
        except Exception as e:
            print(f"Error processing URL {url}: {e}")
            return []

    def _chunk_text(
        self,
        text: str,
        source: str,
        chunk_size: int = 1000,
        overlap: int = 200,
    ) -> list[dict[str, str]]:
        """Splits text into chunks."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            if end >= text_len:
                chunk = text[start:]
            else:
                # Try to find the last period or newline to break cleanly
                last_period = text.rfind(".", start, end)
                last_newline = text.rfind("\n", start, end)
                break_point = max(last_period, last_newline)

                if break_point != -1 and break_point > start + chunk_size // 2:
                    end = break_point + 1  # Include the punctuation

                chunk = text[start:end]

            normalized_chunk = chunk.strip()
            if normalized_chunk:
                chunks.append({"text": normalized_chunk, "source": source})
            start = end - overlap if end < text_len else text_len

        return chunks

