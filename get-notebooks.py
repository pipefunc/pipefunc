# /// script
# dependencies = [
#   "jupytext",
#   "PyGithub",
#   "aiohttp",
# ]
# ///

import asyncio
import os
from pathlib import Path

import aiohttp
from github import Github  # PyGithub


async def download_file_async(session, url, destination) -> None:
    """Downloads a file asynchronously."""
    async with session.get(url) as response:
        response.raise_for_status()
        with open(destination, "wb") as file:
            async for chunk in response.content.iter_chunked(8192):
                file.write(chunk)
        print(f"Downloaded: {url} -> {destination}")


async def convert_to_ipynb_async(md_file, ipynb_file) -> None:
    """Converts a Markdown file to a Jupyter Notebook asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        "jupytext",
        "--to",
        "ipynb",
        md_file,
        "-o",
        ipynb_file,
    )
    await proc.wait()
    if proc.returncode == 0:
        print(f"Converted: {md_file} -> {ipynb_file}")
    else:
        print(f"Error converting {md_file}: Exit code {proc.returncode}")


async def process_file(session, github_file, output_base_dir) -> None:
    """Downloads and converts a single file."""
    file_url = github_file.download_url
    # Extract relevant part of the path, excluding 'docs/source/'
    sub_path = github_file.path.replace("docs/source/", "")
    folder = sub_path.split("/")[0]

    md_file = os.path.join(output_base_dir, sub_path)
    ipynb_file = os.path.join(
        output_base_dir,
        folder,
        Path(github_file.name).stem + ".ipynb",
    )

    # Create intermediate directories for the .md file
    os.makedirs(os.path.dirname(md_file), exist_ok=True)

    await download_file_async(session, file_url, md_file)
    await convert_to_ipynb_async(md_file, ipynb_file)
    os.remove(md_file)  # Remove the temporary .md file


async def main() -> None:
    """Downloads Markdown files from GitHub and converts them to Jupyter Notebooks."""
    repo_name = "pipefunc/pipefunc"  # Your GitHub repository name
    folders = ["examples", "concepts"]
    output_base_dir = "notebooks"
    access_token = os.environ.get(
        "GITHUB_TOKEN",
    )  # Highly recommended to set this environment variable

    # Create output directory structure
    os.makedirs(output_base_dir, exist_ok=True)
    for folder in folders:
        os.makedirs(os.path.join(output_base_dir, folder), exist_ok=True)

    # GitHub API setup
    if access_token:
        g = Github(access_token)
    else:
        g = Github()  # Warning: if unauthenticated, you might hit rate limits
    repo = g.get_repo(repo_name)

    # Download and convert files concurrently
    async with aiohttp.ClientSession() as session:
        tasks = []
        for folder in folders:
            contents = repo.get_contents(f"docs/source/{folder}")
            for file in contents:
                if file.name.endswith(".md"):
                    tasks.append(process_file(session, file, output_base_dir))

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
