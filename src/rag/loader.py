"""
Document Loader for Policy Documents.
Implements Lab 3 (Document Loading & Metadata Extraction).
Supports PyMuPDF (fitz) and pypdf with page-level metadata tracking.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class LoadedPage:
    """Represents a single extracted page with metadata."""
    def __init__(self, doc_name: str, page_number: int, text: str, metadata: Optional[Dict[str, Any]] = None):
        self.doc_name = doc_name
        self.page_number = page_number
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"<LoadedPage doc={self.doc_name} page={self.page_number} chars={len(self.text)}>"


class PolicyDocumentLoader:
    """Loads PDF documents from a directory, preserving page numbers and document titles."""

    def __init__(self, doc_dir: Optional[Path] = None):
        if doc_dir is None:
            self.doc_dir = Path(__file__).resolve().parent.parent.parent / "data" / "policy_documents"
        else:
            self.doc_dir = Path(doc_dir)

    def load_pdf(self, pdf_path: Path) -> List[LoadedPage]:
        """Extracts text page-by-page from a single PDF document."""
        pages: List[LoadedPage] = []
        doc_name = pdf_path.name

        # Method 1: Try PyMuPDF
        try:
            import pymupdf  # Modern import
            doc = pymupdf.open(str(pdf_path))
            for page_idx, page in enumerate(doc):
                text = page.get_text()
                if text and text.strip():
                    pages.append(
                        LoadedPage(
                            doc_name=doc_name,
                            page_number=page_idx + 1,
                            text=text.strip(),
                            metadata={
                                "file_path": str(pdf_path),
                                "total_pages": len(doc),
                                "source_engine": "pymupdf"
                            }
                        )
                    )
            doc.close()
            if pages:
                return pages
        except Exception as e:
            logger.debug(f"PyMuPDF extraction failed for {doc_name}: {e}. Trying fallback.")

        # Method 2: Fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(pdf_path))
            for page_idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text and text.strip():
                    pages.append(
                        LoadedPage(
                            doc_name=doc_name,
                            page_number=page_idx + 1,
                            text=text.strip(),
                            metadata={
                                "file_path": str(pdf_path),
                                "total_pages": len(reader.pages),
                                "source_engine": "pypdf"
                            }
                        )
                    )
        except Exception as e:
            logger.warning(f"pypdf extraction failed for {doc_name}: {e}")

        return pages

    def load_all_documents(self) -> List[LoadedPage]:
        """Loads all available PDF documents from the policy_documents directory."""
        if not self.doc_dir.exists():
            logger.warning(f"Policy directory {self.doc_dir} does not exist.")
            return []

        all_pages: List[LoadedPage] = []
        pdf_files = sorted(list(self.doc_dir.glob("*.pdf")))
        logger.info(f"Found {len(pdf_files)} PDF files in {self.doc_dir}")

        for pdf_path in pdf_files:
            pages = self.load_pdf(pdf_path)
            logger.info(f"Loaded {len(pages)} text pages from {pdf_path.name}")
            all_pages.extend(pages)

        return all_pages
