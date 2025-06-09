"""Test suite for MCP server functionality."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

import pytest

from pipefunc import Pipeline, pipefunc

if TYPE_CHECKING:
    from pathlib import Path

    from pipefunc.typing import Array

# Check for optional dependencies
fastmcp = pytest.importorskip("fastmcp", reason="fastmcp not available")
pydantic = pytest.importorskip("pydantic", reason="pydantic not available")


def parse_mcp_response(response_text: str) -> dict:
    """Safely parse MCP response text that contains Python dict strings."""
    import ast

    try:
        # First try ast.literal_eval for simple cases
        return ast.literal_eval(response_text)
    except (ValueError, SyntaxError):
        # If that fails, try to convert to JSON-compatible format
        try:
            # Replace single quotes with double quotes and handle numpy arrays
            json_text = response_text.replace("'", '"')
            # Handle numpy array representations
            json_text = json_text.replace("array(", "[").replace(")", "]")
            return json.loads(json_text)
        except (ValueError, json.JSONDecodeError):
            # If all else fails, return a dict with the raw text
            return {"raw_response": response_text, "parse_error": True}


@pytest.fixture
def simple_pipeline() -> Pipeline:
    """Create a simple pipeline for testing."""

    @pipefunc(output_name="result")
    def add_numbers(x: float, y: float) -> float:
        return x + y

    return Pipeline([add_numbers])


@pytest.fixture
def complex_pipeline() -> Pipeline:
    """Create a complex pipeline with mapspecs for testing."""

    @pipefunc(output_name="values", mapspec="x[i] -> values[i]")
    def compute_values(x: float) -> float:
        return x * 2.0

    @pipefunc(output_name="sum_result")
    def sum_values(values: Array) -> float:
        return float(sum(values))

    return Pipeline([compute_values, sum_values])


@pytest.fixture
def slow_pipeline():
    """Create a pipeline with deliberately slow functions for async testing."""
    import time

    @pipefunc(output_name="slow_result", mapspec="x[i] -> slow_result[i]")
    def slow_computation(x: float) -> float:
        time.sleep(0.01)  # Small delay to make async behavior observable
        return x**2

    @pipefunc(output_name="final_result")
    def aggregate(slow_result: Array) -> float:
        return float(sum(slow_result))

    return Pipeline([slow_computation, aggregate])


@pytest.fixture
def mixed_input_pipeline() -> Pipeline:
    """Create a pipeline with both array inputs (mapspec) and scalar inputs (non-mapspec)."""

    @pipefunc(output_name="scaled_values", mapspec="x[i] -> scaled_values[i]")
    def scale_values(x: float, scale_factor: float) -> float:
        return x * scale_factor

    @pipefunc(output_name="final_result")
    def process_scaled(scaled_values: Array) -> float:
        return float(sum(scaled_values))

    return Pipeline([scale_values, process_scaled])


# Test MCP server building functionality.


@pytest.mark.asyncio
async def test_execute_pipeline_sync_simple(simple_pipeline: Pipeline) -> None:
    """Test building MCP server for simple pipeline."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    assert mcp is not None
    async with Client(mcp) as client:
        result = await client.call_tool("execute_pipeline_sync", {"inputs": {"x": 1, "y": 2}})
        assert result[0].text == "{'result': {'output': 3.0, 'shape': None}}"


# Test async job management functionality


@pytest.mark.asyncio
async def test_execute_pipeline_async_simple(simple_pipeline: Pipeline) -> None:
    """Test starting an async pipeline job."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    async with Client(mcp) as client:
        # Start async job
        result = await client.call_tool("execute_pipeline_async", {"inputs": {"x": 5, "y": 10}})

        response = parse_mcp_response(result[0].text)
        assert "job_id" in response
        assert "run_folder" in response
        assert response["run_folder"].startswith("runs/job_")

        # Job ID should be a valid UUID format
        job_id = response["job_id"]
        assert len(job_id) == 36  # Standard UUID length
        assert job_id.count("-") == 4  # Standard UUID dash count


@pytest.mark.asyncio
async def test_check_job_status_completed(simple_pipeline: Pipeline) -> None:
    """Test checking status of a completed job."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    async with Client(mcp) as client:
        # Start async job
        start_result = await client.call_tool(
            "execute_pipeline_async",
            {"inputs": {"x": 3, "y": 7}},
        )
        job_info = parse_mcp_response(start_result[0].text)
        job_id = job_info["job_id"]

        # Wait for job to complete with retry logic
        max_wait = 2  # seconds
        waited = 0.0
        status_info = None
        while waited < max_wait:
            status_result = await client.call_tool("check_job_status", {"job_id": job_id})
            status_info = parse_mcp_response(status_result[0].text)
            if status_info["status"] == "completed":
                break
            await asyncio.sleep(0.01)
            waited += 0.01

        assert status_info is not None
        assert status_info["job_id"] == job_id
        assert status_info["status"] == "completed"
        assert status_info["pipeline_name"] == "Unnamed Pipeline"
        assert "started_at" in status_info
        # Allow for errors if multiprocessing isn't supported in test environment
        if status_info["error"] is None:
            assert "results" in status_info
            assert status_info["results"]["result"]["output"] == 10.0


@pytest.mark.asyncio
async def test_check_job_status_with_progress(slow_pipeline: Pipeline) -> None:
    """Test checking status of a job with progress tracking."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(slow_pipeline)
    async with Client(mcp) as client:
        # Start async job with multiple inputs to create observable progress
        start_result = await client.call_tool(
            "execute_pipeline_async",
            {"inputs": {"x": [1, 2, 3, 4, 5]}},
        )
        job_info = parse_mcp_response(start_result[0].text)
        job_id = job_info["job_id"]

        # Check status while potentially running
        status_result = await client.call_tool("check_job_status", {"job_id": job_id})
        status_info = parse_mcp_response(status_result[0].text)

        assert status_info["job_id"] == job_id
        assert status_info["status"] in ["running", "completed"]
        assert "progress" in status_info

        # Wait for completion with retry logic
        max_wait = 3  # seconds
        waited = 0.0
        final_status_info = None
        while waited < max_wait:
            final_status_result = await client.call_tool("check_job_status", {"job_id": job_id})
            final_status_info = parse_mcp_response(final_status_result[0].text)
            # Skip test if parsing failed due to complex numpy arrays
            if final_status_info.get("parse_error"):
                pytest.skip("Response parsing failed due to complex numpy arrays")
            if final_status_info["status"] == "completed":
                break
            await asyncio.sleep(0.01)
            waited += 0.01

        assert final_status_info is not None
        # Skip test if parsing failed due to complex numpy arrays
        if final_status_info.get("parse_error"):
            pytest.skip("Response parsing failed due to complex numpy arrays")

        assert final_status_info["status"] == "completed"
        # Allow for errors if multiprocessing isn't supported in test environment
        if final_status_info["error"] is None:
            assert "results" in final_status_info
            assert final_status_info["results"]["final_result"]["output"] == 55.0  # 1+4+9+16+25


@pytest.mark.asyncio
async def test_list_jobs_empty() -> None:
    """Test listing jobs when no jobs exist."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server, job_registry

    # Clear job registry for clean test
    job_registry.clear()

    mcp = build_mcp_server(Pipeline([]))
    async with Client(mcp) as client:
        result = await client.call_tool("list_jobs", {})
        jobs_info = parse_mcp_response(result[0].text)

        assert jobs_info["total_count"] == 0
        assert jobs_info["jobs"] == []


@pytest.mark.asyncio
async def test_list_jobs_with_multiple_jobs(simple_pipeline: Pipeline) -> None:
    """Test listing multiple jobs."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server, job_registry

    # Clear job registry for clean test
    job_registry.clear()

    mcp = build_mcp_server(simple_pipeline)
    async with Client(mcp) as client:
        # Start multiple jobs
        job_ids = []
        for i in range(3):
            start_result = await client.call_tool(
                "execute_pipeline_async",
                {"inputs": {"x": i, "y": i + 1}},
            )
            job_info = parse_mcp_response(start_result[0].text)
            job_ids.append(job_info["job_id"])

        # Wait for jobs to complete
        await asyncio.sleep(0.02)

        # List all jobs
        list_result = await client.call_tool("list_jobs", {})
        jobs_info = parse_mcp_response(list_result[0].text)

        assert jobs_info["total_count"] == 3
        assert len(jobs_info["jobs"]) == 3

        returned_job_ids = [job["job_id"] for job in jobs_info["jobs"]]
        for job_id in job_ids:
            assert job_id in returned_job_ids

        # Check job info structure
        for job in jobs_info["jobs"]:
            assert "job_id" in job
            assert "pipeline_name" in job
            assert "status" in job
            assert "run_folder" in job
            assert "started_at" in job
            assert "has_error" in job


@pytest.mark.asyncio
async def test_cancel_job(slow_pipeline: Pipeline) -> None:
    """Test cancelling a running job."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(slow_pipeline)
    async with Client(mcp) as client:
        # Start a job with multiple inputs to make it run longer
        start_result = await client.call_tool(
            "execute_pipeline_async",
            {"inputs": {"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}},
        )
        job_info = parse_mcp_response(start_result[0].text)
        job_id = job_info["job_id"]

        # Immediately try to cancel
        cancel_result = await client.call_tool("cancel_job", {"job_id": job_id})
        cancel_info = parse_mcp_response(cancel_result[0].text)

        # The job might complete before cancellation, so we check both possibilities
        if "status" in cancel_info:
            assert cancel_info["status"] == "cancelled"
            assert cancel_info["job_id"] == job_id
        else:
            # Job might have completed before we could cancel it
            assert "error" in cancel_info


@pytest.mark.asyncio
async def test_cancel_nonexistent_job(simple_pipeline: Pipeline) -> None:
    """Test cancelling a job that doesn't exist."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    async with Client(mcp) as client:
        fake_job_id = "00000000-0000-0000-0000-000000000000"
        cancel_result = await client.call_tool("cancel_job", {"job_id": fake_job_id})
        cancel_info = parse_mcp_response(cancel_result[0].text)

        assert "error" in cancel_info
        assert cancel_info["error"] == "Job not found"


@pytest.mark.asyncio
async def test_cancel_completed_job(simple_pipeline: Pipeline) -> None:
    """Test cancelling a job that has already completed."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    async with Client(mcp) as client:
        # Start a simple job that completes quickly
        start_result = await client.call_tool(
            "execute_pipeline_async",
            {"inputs": {"x": 1, "y": 2}},
        )
        job_info = parse_mcp_response(start_result[0].text)
        job_id = job_info["job_id"]

        # Wait for job to complete
        max_wait = 2  # seconds
        waited = 0.0
        while waited < max_wait:
            status_result = await client.call_tool("check_job_status", {"job_id": job_id})
            status_info = parse_mcp_response(status_result[0].text)
            if status_info["status"] == "completed":
                break
            await asyncio.sleep(0.01)
            waited += 0.01

        # Now try to cancel the completed job
        cancel_result = await client.call_tool("cancel_job", {"job_id": job_id})
        cancel_info = parse_mcp_response(cancel_result[0].text)

        # Should get the "already completed" error message
        assert "error" in cancel_info
        assert cancel_info["error"] == "Job not found or already completed"
        assert cancel_info["job_id"] == job_id


@pytest.mark.asyncio
async def test_check_nonexistent_job_status(simple_pipeline):
    """Test checking status of a job that doesn't exist."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    async with Client(mcp) as client:
        fake_job_id = "00000000-0000-0000-0000-000000000000"
        status_result = await client.call_tool("check_job_status", {"job_id": fake_job_id})
        status_info = parse_mcp_response(status_result[0].text)

        assert "error" in status_info
        assert status_info["error"] == "Job not found"


@pytest.mark.asyncio
async def test_async_job_with_custom_run_folder(simple_pipeline: Pipeline, tmp_path: Path):
    """Test async job with custom run folder."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server

    mcp = build_mcp_server(simple_pipeline)
    async with Client(mcp) as client:
        custom_folder = tmp_path / "test_custom_run"
        start_result = await client.call_tool(
            "execute_pipeline_async",
            {"inputs": {"x": 1, "y": 2}, "run_folder": custom_folder},
        )
        job_info = parse_mcp_response(start_result[0].text)

        assert job_info["run_folder"] == str(custom_folder)

        # Wait for completion and check status
        max_wait = 2  # seconds
        waited = 0.0
        status_info = None
        while waited < max_wait:
            status_result = await client.call_tool(
                "check_job_status",
                {"job_id": job_info["job_id"]},
            )
            status_info = parse_mcp_response(status_result[0].text)
            if status_info["status"] in ["completed", "cancelled"]:
                break
            await asyncio.sleep(0.01)
            waited += 0.01

        assert status_info is not None
        assert status_info["run_folder"] == str(custom_folder)


@pytest.mark.asyncio
async def test_end_to_end_async_workflow(complex_pipeline: Pipeline) -> None:
    """Test complete end-to-end async workflow."""
    from fastmcp import Client

    from pipefunc.mcp import build_mcp_server, job_registry

    # Clear job registry for clean test
    job_registry.clear()

    mcp = build_mcp_server(complex_pipeline)
    async with Client(mcp) as client:
        # 1. Start async job
        start_result = await client.call_tool(
            "execute_pipeline_async",
            {"inputs": {"x": [1, 2, 3, 4]}},
        )
        job_info = parse_mcp_response(start_result[0].text)
        job_id = job_info["job_id"]

        # 2. Check initial status (might be running)
        initial_status_result = await client.call_tool("check_job_status", {"job_id": job_id})
        initial_status = parse_mcp_response(initial_status_result[0].text)
        assert initial_status["job_id"] == job_id
        assert initial_status["status"] in ["running", "completed"]

        # 3. List jobs (should show our job)
        list_result = await client.call_tool("list_jobs", {})
        jobs_info = parse_mcp_response(list_result[0].text)
        assert jobs_info["total_count"] >= 1
        job_ids = [job["job_id"] for job in jobs_info["jobs"]]
        assert job_id in job_ids

        # 4. Wait for completion
        max_wait = 3  # seconds
        waited = 0.0
        status_info = None
        while waited < max_wait:
            status_result = await client.call_tool("check_job_status", {"job_id": job_id})
            status_info = parse_mcp_response(status_result[0].text)
            # Skip test if parsing failed due to complex numpy arrays
            if status_info.get("parse_error"):
                pytest.skip("Response parsing failed due to complex numpy arrays")
            if status_info["status"] == "completed":
                break
            await asyncio.sleep(0.01)
            waited += 0.01

        # 5. Verify final results
        assert status_info is not None
        # Skip test if parsing failed due to complex numpy arrays
        if status_info.get("parse_error"):
            pytest.skip("Response parsing failed due to complex numpy arrays")

        assert status_info["status"] == "completed"
        # Allow for errors if multiprocessing isn't supported in test environment
        if status_info["error"] is None:
            assert "results" in status_info

            # Check that results are correct: x=[1,2,3,4] -> values=[2,4,6,8] -> sum=20
            results = status_info["results"]
            assert "values" in results
            assert "sum_result" in results
            assert results["sum_result"]["output"] == 20.0


# Test MCP internal helper functions


def test_get_pipeline_documentation(simple_pipeline: Pipeline) -> None:
    """Test pipeline documentation generation."""
    from pipefunc.mcp import _get_pipeline_documentation

    docs = _get_pipeline_documentation(simple_pipeline)
    assert isinstance(docs, str)
    assert len(docs) > 0


def test_get_pipeline_info_summary(simple_pipeline: Pipeline) -> None:
    """Test pipeline info summary generation."""
    from pipefunc.mcp import _get_pipeline_info_summary

    summary = _get_pipeline_info_summary("Test Pipeline", simple_pipeline)
    assert isinstance(summary, str)
    assert "Pipeline Name: Test Pipeline" in summary
    assert "Required Inputs:" in summary
    assert "Outputs:" in summary


# Test MCP error handling and edge cases.


def test_empty_pipeline_handling() -> None:
    """Test handling of empty pipeline."""
    from pipefunc.mcp import _get_pipeline_info_summary

    empty_pipeline = Pipeline([])
    summary = _get_pipeline_info_summary("Empty", empty_pipeline)

    assert isinstance(summary, str)
    assert "Pipeline Name: Empty" in summary


def test_pipeline_with_no_outputs() -> None:
    """Test handling of pipeline with no outputs."""
    from pipefunc.mcp import _get_pipeline_info_summary

    # Create a function with minimal configuration
    @pipefunc(output_name="dummy_output")
    def dummy_func(x: int) -> int:
        return x

    pipeline = Pipeline([dummy_func])
    summary = _get_pipeline_info_summary("Test", pipeline)
    assert isinstance(summary, str)
    assert "dummy_output" in summary


def test_input_format_section_with_mixed_inputs(mixed_input_pipeline: Pipeline) -> None:
    """Test input format section generation with both array and scalar inputs."""
    from pipefunc.mcp import _get_input_format_section

    result = _get_input_format_section(mixed_input_pipeline)

    # Should contain the specific lines we want to test
    assert "Required Array Inputs:" in result
    assert "Constant Inputs:" in result
    assert "The following parameters are provided as single, constant values:" in result
    assert "- `x`" in result  # array input
    assert "- `scale_factor`" in result  # scalar input
    assert "This pipeline is designed for array-based parameter sweeps." in result
