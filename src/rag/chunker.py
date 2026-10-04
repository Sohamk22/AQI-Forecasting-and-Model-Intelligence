"""
Document Chunker for Policy Documents.
Implements Lab 3 (Chunking & Source Traceability).
Splits document pages into semantically cohesive chunks while preserving metadata.
"""
import re
from typing import List, Optional
from src.rag.loader import LoadedPage
from src.rag.schemas import DocumentChunk


class PolicyChunker:
    """
    Chunks extracted pages using sliding window character/word boundaries
    while strictly preserving document name, page number, and offset metadata.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_length: int = 50
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

    def _split_into_paragraphs_or_sentences(self, text: str) -> List[str]:
        """Splits raw page text into natural paragraph and sentence units."""
        # Normalize double newlines and carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Split on paragraph breaks or multi-newlines
        raw_paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        
        units: List[str] = []
        for p in raw_paragraphs:
            if len(p) <= self.chunk_size:
                units.append(p)
            else:
                # Split large paragraphs by sentence endings
                sentences = re.split(r"(?<=[.!?])\s+", p)
                current_unit = ""
                for s in sentences:
                    if len(current_unit) + len(s) + 1 <= self.chunk_size:
                        current_unit = f"{current_unit} {s}".strip()
                    else:
                        if current_unit:
                            units.append(current_unit)
                        current_unit = s
                if current_unit:
                    units.append(current_unit)
        return units

    def chunk_page(self, page: LoadedPage) -> List[DocumentChunk]:
        """Splits a single LoadedPage into a sequence of DocumentChunk objects."""
        text = page.text
        if not text or len(text.strip()) < self.min_chunk_length:
            return []

        units = self._split_into_paragraphs_or_sentences(text)
        chunks: List[DocumentChunk] = []
        
        current_chunk_text = ""
        chunk_idx = 0
        char_offset = 0

        for unit in units:
            if not current_chunk_text:
                current_chunk_text = unit
            elif len(current_chunk_text) + len(unit) + 1 <= self.chunk_size:
                current_chunk_text = f"{current_chunk_text}\n{unit}"
            else:
                # Save current chunk
                chunk_text = current_chunk_text.strip()
                if len(chunk_text) >= self.min_chunk_length:
                    chunk_id = f"{page.doc_name}:p{page.page_number}:c{chunk_idx}"
                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            doc_name=page.doc_name,
                            page_number=page.page_number,
                            text=chunk_text,
                            char_start=char_offset,
                            char_end=char_offset + len(chunk_text),
                            metadata=page.metadata.copy()
                        )
                    )
                    chunk_idx += 1
                    char_offset += len(chunk_text)

                # Apply overlap: keep the tail of the current chunk
                overlap_text = current_chunk_text[-self.chunk_overlap:] if len(current_chunk_text) > self.chunk_overlap else ""
                current_chunk_text = f"{overlap_text}\n{unit}".strip()

        # Append trailing chunk
        if current_chunk_text and len(current_chunk_text.strip()) >= self.min_chunk_length:
            chunk_text = current_chunk_text.strip()
            chunk_id = f"{page.doc_name}:p{page.page_number}:c{chunk_idx}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_name=page.doc_name,
                    page_number=page.page_number,
                    text=chunk_text,
                    char_start=char_offset,
                    char_end=char_offset + len(chunk_text),
                    metadata=page.metadata.copy()
                )
            )

        return chunks

    def chunk_documents(self, pages: List[LoadedPage]) -> List[DocumentChunk]:
        """Chunks a collection of loaded pages across multiple documents."""
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            page_chunks = self.chunk_page(page)
            all_chunks.extend(page_chunks)
        return all_chunks
