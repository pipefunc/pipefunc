import os
import subprocess
from pathlib import Path

import requests


def download_file(url, destination):
    """Downloads a file from a URL to a local destination."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    with open(destination, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)


def convert_to_ipynb(md_file, ipynb_file):
    """Converts a Markdown file to a Jupyter Notebook using jupytext."""
    try:
        subprocess.run(["jupytext", "--to", "ipynb", md_file, "-o", ipynb_file], check=True)
        print(f"Converted: {md_file} -> {ipynb_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting {md_file}: {e}")


def main():
    """Downloads Markdown files from GitHub and converts them to Jupyter Notebooks."""
    repo_url = "https://raw.githubusercontent.com/pipefunc/pipefunc/main/docs/source/"  # Base URL of your repo's raw files
    folders = ["examples", "concepts"]
    output_base_dir = "notebooks"  # Name of the output directory

    # Create the output directory structure
    os.makedirs(output_base_dir, exist_ok=True)
    for folder in folders:
        os.makedirs(os.path.join(output_base_dir, folder), exist_ok=True)

    # Iterate through folders and download/convert files
    for folder in folders:
        folder_url = os.path.join(repo_url, folder)
        # Get the list of .md files from the GitHub directory listing (simulated)
        # In reality, you'd need to fetch the directory listing, but we're using a direct approach for simplicity
        if folder == "examples":
            md_files = [
                "basic-usage.md",
                "image-processing.md",
                "index.md",
                "nlp-text-summarization.md",
                "physics-simulation.md",
                "sensor-data-processing.md",
                "weather-simulation.md",
            ]
        elif folder == "concepts":
            md_files = [
                "adaptive-integration.md",
                "error-handling.md",
                "execution-and-parallelism.md",
                "function-io.md",
                "index.md",
                "mapspec.md",
                "overhead-and-efficiency.md",
                "parameter-scopes.md",
                "parameter-sweeps.md",
                "resource-management.md",
                "simplifying-pipelines.md",
                "testing.md",
                "type-checking.md",
                "variants.md",
            ]
        else:
            md_files = []

        for md_file in md_files:
            file_url = os.path.join(folder_url, md_file)
            md_filepath = os.path.join(output_base_dir, folder, md_file)
            ipynb_filepath = os.path.join(output_base_dir, folder, Path(md_file).stem + ".ipynb")

            try:
                download_file(file_url, md_filepath)
                print(f"Downloaded: {file_url} -> {md_filepath}")
                convert_to_ipynb(md_filepath, ipynb_filepath)
                # Optionally remove the downloaded Markdown file if you only need the .ipynb
                os.remove(md_filepath)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file_url}: {e}")


if __name__ == "__main__":
    main()
