"""
Pydantic Schemas for AeroPulse Policy Intelligence RAG Layer.
Implements Lab 1 (Structured Outputs) and Lab 4 (Grounded Evidence Schemas).
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Document & Ingestion Schemas
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """Represents a discrete chunk of text extracted from a policy document."""
    chunk_id: str = Field(..., description="Unique identifier: doc_name:page:chunk_index")
    doc_name: str = Field(..., description="Source PDF filename or document title")
    page_number: int = Field(..., description="1-indexed page number from the source document")
    text: str = Field(..., description="The raw textual content of the chunk")
    char_start: int = Field(0, description="Character offset start in document/page")
    char_end: int = Field(0, description="Character offset end in document/page")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional document metadata")


class RetrievalResult(BaseModel):
    """Result of dense, lexical, or hybrid retrieval."""
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    hybrid_score: float = 0.0
    rank: int = 0


# ---------------------------------------------------------------------------
# 2. Forecast & Policy Context Schemas
# ---------------------------------------------------------------------------

class PriorSummary(BaseModel):
    """Prior 14-day meteorological and pollutant summary (strictly prior-only)."""
    mean_aqi: Optional[float] = None
    mean_pm25: Optional[float] = None
    mean_pm10: Optional[float] = None
    mean_temp: Optional[float] = None
    mean_humidity: Optional[float] = None
    mean_wind: Optional[float] = None


class ForecastPolicyContext(BaseModel):
    """
    Structured context extracted exclusively from prior-only forecast inference.
    NEVER includes contemporaneous (t=0) ground-truth AQI.
    """
    city: str = Field(default="Delhi", description="City location for policy applicability")
    target_date: str = Field(..., description="Target forecast date (YYYY-MM-DD)")
    predicted_aqi: float = Field(..., description="Ensemble consensus or primary predicted AQI")
    predicted_category: str = Field(..., description="CPCB category: Good, Satisfactory, Moderate, Poor, Very Poor, Severe")
    dominant_pollutant: Optional[str] = Field(default="PM2.5", description="Primary pollutant driver")
    health_risk: str = Field(..., description="CPCB associated health impact description")
    model_predictions: Dict[str, float] = Field(default_factory=dict, description="Individual model forecasts")
    prior_14_days_summary: Optional[PriorSummary] = None
    methodology_note: str = Field(
        default="Policy analysis derived strictly from prior-only ML forecast without future leakage.",
        description="Scientific integrity note"
    )


class PolicyQuery(BaseModel):
    """Focused query generated from forecast context."""
    query_text: str = Field(..., description="Natural language search query")
    target_date: str
    predicted_category: str
    severity_level: str
    key_pollutants: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# 3. Grounded Policy Response Schemas (Lab 1 & Lab 4)
# ---------------------------------------------------------------------------

class CitationSource(BaseModel):
    """Verified reference to an official policy document."""
    doc_name: str = Field(..., description="Document filename")
    doc_title: str = Field(default="", description="Human-readable display title of policy document")
    page_number: int = Field(..., description="1-indexed source page")
    chunk_id: str = Field(..., description="Chunk identifier for auditability")
    excerpt: str = Field(..., description="Short verifiable excerpt from the source text")
    relevance_note: Optional[str] = Field(default="", description="Why this evidence supports the statement")


class PolicyMeasure(BaseModel):
    """Specific actionable policy or mitigation measure."""
    category: str = Field(..., description="Sector: e.g. Dust Control, Transport, Industry, Biomass, Emergency")
    measure: str = Field(..., description="Detailed description of mandated or recommended action")
    authority: str = Field(default="Municipal/State Authority", description="Enforcing agency (e.g. NDMC, DPCC, CAQM)")
    urgency: str = Field(default="Standard", description="Low, Moderate, High, Emergency")
    source_citation: CitationSource = Field(..., description="Source citation backing this measure")


class PolicyAnalysis(BaseModel):
    """
    Complete structured policy intelligence report.
    Guaranteed to be grounded strictly in retrieved evidence.
    """
    target_date: str
    forecasted_aqi: float
    forecasted_category: str
    applicable_stage: str = Field(
        ...,
        description="Applicable Graded Action Stage or Mitigation Alert Level based strictly on evidence"
    )
    summary: str = Field(..., description="Executive grounded summary of air quality policy implications")
    relevant_measures: List[PolicyMeasure] = Field(
        default_factory=list,
        description="Actionable measures grounded in official policy documents"
    )
    citations: List[CitationSource] = Field(
        default_factory=list,
        description="All document citations referenced in analysis"
    )
    caveats: List[str] = Field(
        default_factory=list,
        description="Methodological caveats, data limitations, or sufficiency qualifications"
    )
    evidence_sufficiency: str = Field(
        default="Sufficient",
        description="'Sufficient', 'Partial', or 'Insufficient Evidence'"
    )


# ---------------------------------------------------------------------------
# 4. API Request / Response Schemas
# ---------------------------------------------------------------------------

class PolicyAnalyzeRequest(BaseModel):
    """Payload for POST /api/policy/analyze."""
    date: Optional[str] = Field("2020-01-15", description="Target forecast date")
    query_override: Optional[str] = Field(None, description="Optional custom policy question")
    top_k: Optional[int] = Field(5, description="Number of policy chunks to retrieve")


class PolicyAnalyzeResponse(BaseModel):
    """Response returned by POST /api/policy/analyze."""
    status: str = "success"
    context: ForecastPolicyContext
    analysis: PolicyAnalysis
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
