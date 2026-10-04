"""
Policy Context & Query Builder.
Connects the existing AeroPulse ML forecasting outputs to the RAG policy intelligence layer.
Strictly ensures zero future/actual AQI leakage into policy reasoning.
"""
from typing import Dict, Any, Optional
from src.rag.schemas import ForecastPolicyContext, PolicyQuery, PriorSummary


class PolicyContextBuilder:
    """
    Transforms ML inference forecast outputs into a leak-free ForecastPolicyContext
    and generates an optimized retrieval query.
    """

    @staticmethod
    def from_forecast_payload(
        forecast_data: Dict[str, Any],
        city: str = "Delhi"
    ) -> ForecastPolicyContext:
        """
        Builds a ForecastPolicyContext strictly from forecast/predicted fields and
        prior 14-day history. NEVER extracts or relies on actual_aqi.
        """
        target_date = forecast_data.get("target_date", "2020-01-15")
        
        # Primary consensus or top model prediction
        consensus_aqi = forecast_data.get("consensus_aqi")
        if consensus_aqi is None:
            # Fallback to individual models
            xgb_pred = forecast_data.get("XGBoost", {}).get("predicted_aqi")
            se1_pred = forecast_data.get("SE-1", {}).get("predicted_aqi")
            se2_pred = forecast_data.get("SE-2", {}).get("predicted_aqi")
            valid_preds = [p for p in [xgb_pred, se1_pred, se2_pred] if p is not None]
            consensus_aqi = float(sum(valid_preds) / len(valid_preds)) if valid_preds else 200.0

        # Model individual predictions dictionary
        model_preds = {}
        for m in ["XGBoost", "SE-1", "SE-2"]:
            if m in forecast_data and "predicted_aqi" in forecast_data[m]:
                model_preds[m] = float(forecast_data[m]["predicted_aqi"])

        # Predicted category and health risk from consensus or primary model
        primary_model = forecast_data.get("SE-1") or forecast_data.get("XGBoost") or {}
        category = primary_model.get("category", "Moderate")
        health_risk = primary_model.get("health_risk", "Breathing discomfort to sensitive people")

        # Extract prior 14 days summary (legitimately available before forecast date)
        prior_raw = forecast_data.get("prior_14_days_summary", {})
        prior_summary = None
        if prior_raw:
            prior_summary = PriorSummary(
                mean_aqi=prior_raw.get("mean_aqi"),
                mean_pm25=prior_raw.get("mean_pm25"),
                mean_pm10=prior_raw.get("mean_pm10"),
                mean_temp=prior_raw.get("mean_temp"),
                mean_humidity=prior_raw.get("mean_humidity"),
                mean_wind=prior_raw.get("mean_wind")
            )

        # Detect dominant pollutant from prior means or defaults
        dominant = "PM2.5"
        if prior_summary and prior_summary.mean_pm10 and prior_summary.mean_pm25:
            if prior_summary.mean_pm10 > prior_summary.mean_pm25 * 2.5:
                dominant = "PM10 (Coarse Dust)"
            else:
                dominant = "PM2.5 (Combustion/Particulate)"

        return ForecastPolicyContext(
            city=city,
            target_date=target_date,
            predicted_aqi=round(float(consensus_aqi), 1),
            predicted_category=category,
            dominant_pollutant=dominant,
            health_risk=health_risk,
            model_predictions=model_preds,
            prior_14_days_summary=prior_summary,
            methodology_note="Policy context generated exclusively from t-14 to t-1 prior forecast inference. Zero contemporaneous/actual AQI leakage."
        )

    @staticmethod
    def build_policy_query(
        context: ForecastPolicyContext,
        query_override: Optional[str] = None
    ) -> PolicyQuery:
        """
        Generates an evidence-seeking query from forecast parameters without pre-judging outcomes.
        """
        if query_override and query_override.strip():
            return PolicyQuery(
                query_text=query_override.strip(),
                target_date=context.target_date,
                predicted_category=context.predicted_category,
                severity_level=context.predicted_category,
                key_pollutants=[context.dominant_pollutant or "PM2.5"]
            )

        # Build conceptual evidence query
        aqi_val = context.predicted_aqi
        cat = context.predicted_category
        city = context.city

        # Formulate query based on atmospheric conditions
        if aqi_val > 400 or cat == "Severe":
            keywords = "Severe emergency air pollution GRAP Stage IV Odd-Even anti-smog diesel vehicle ban industrial shutdown crop residue NDMC CAQM"
        elif aqi_val > 300 or cat == "Very Poor":
            keywords = "Very Poor air quality GRAP Stage III construction ban diesel generator mechanized sweeping dust suppression CAQM NDMC"
        elif aqi_val > 200 or cat == "Moderate" or cat == "Poor":
            keywords = "Moderate Poor air quality GRAP Stage I Stage II water sprinkling road dust mechanized sweeping open burning ban"
        else:
            keywords = "Good Satisfactory baseline ambient air quality standards CPCB monitoring green war room compliance"

        query_text = (
            f"Mandated air pollution mitigation measures, municipal action plans, and statutory directives "
            f"applicable to {cat} air quality conditions ({keywords}) in {city} NCR."
        )

        return PolicyQuery(
            query_text=query_text,
            target_date=context.target_date,
            predicted_category=cat,
            severity_level=cat,
            key_pollutants=[context.dominant_pollutant or "PM2.5"]
        )
