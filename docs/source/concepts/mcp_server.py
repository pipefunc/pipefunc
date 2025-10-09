#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["pipefunc[mcp]"]
# ///

"""MCP server that exposes the Series Analyzer pipeline."""

from math import isnan
from statistics import mean, median, pstdev

from pipefunc import Pipeline, pipefunc
from pipefunc.mcp import build_mcp_server


@pipefunc(output_name="clean_series")
def clean_series(series: list[float]) -> list[float]:
    """Remove null and NaN readings before analysis.

    Parameters
    ----------
    series : list[float]
        Raw numeric samples to analyze.

    Returns
    -------
    list[float]
        Cleaned numeric values with missing entries removed.

    """
    cleaned: list[float] = []
    for value in series:
        if value is None:
            continue
        number = float(value)
        if isnan(number):
            continue
        cleaned.append(number)
    if not cleaned:
        msg = "series must contain at least one numeric value"
        raise ValueError(msg)
    return cleaned


@pipefunc(output_name="summary")
def summarize(clean_series: list[float]) -> dict[str, float]:
    """Compute descriptive statistics for the cleaned samples.

    Parameters
    ----------
    clean_series : list[float]
        Sanitized numeric samples.

    Returns
    -------
    dict[str, float]
        Aggregate metrics such as count, mean, median, standard deviation, and range.

    """
    stats = {
        "count": len(clean_series),
        "mean": mean(clean_series),
        "median": median(clean_series),
        "min": min(clean_series),
        "max": max(clean_series),
        "std": pstdev(clean_series) if len(clean_series) > 1 else 0.0,
    }
    stats["range"] = stats["max"] - stats["min"]
    return stats


@pipefunc(output_name="anomalies")
def detect_anomalies(
    clean_series: list[float],
    summary: dict[str, float],
    z_threshold: float = 2.5,
) -> list[dict[str, float]]:
    """Flag values whose z-score exceeds the configured threshold.

    Parameters
    ----------
    clean_series : list[float]
        Sanitized numeric samples.
    summary : dict[str, float]
        Descriptive statistics from :func:`summarize`.
    z_threshold : float, default 2.5
        Absolute z-score required to mark a value as an anomaly.

    Returns
    -------
    list[dict[str, float]]
        Each anomaly with its index, value, and z-score.

    """
    std = summary.get("std", 0.0)
    if std <= 0:
        return []
    mean_value = summary["mean"]
    anomalies: list[dict[str, float]] = []
    for index, value in enumerate(clean_series):
        z_score = (value - mean_value) / std
        if abs(z_score) >= z_threshold:
            anomalies.append({"index": index, "value": value, "z_score": z_score})
    return anomalies


pipeline = Pipeline(
    [clean_series, summarize, detect_anomalies],
    name="Series Analyzer",
)
mcp = build_mcp_server(pipeline)

if __name__ == "__main__":
    # Start server on stdio for agent integration
    mcp.run(path="/series", transport="stdio")
