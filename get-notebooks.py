"""Downloads Markdown files from a GitHub repository and converts them to Jupyter Notebooks."""

# /// script
# dependencies = [
# "jupytext",
# "PyGithub",
# "aiohttp",
# ]
# ///
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
from github import ContentFile, Github  # PyGithub

if TYPE_CHECKING:
    from github.Repository import Repository


async def download_file_async(session: aiohttp.ClientSession, url: str, destination: Path) -> None:
    """Downloads a file asynchronously."""
    async with session.get(url) as response:
        response.raise_for_status()
        with destination.open("wb") as file:
            async for chunk in response.content.iter_chunked(8192):
                file.write(chunk)
        print(f"Downloaded: {url} -> {destination}")


async def convert_to_ipynb_async(md_file: Path, ipynb_file: Path) -> None:
    """Converts a Markdown file to a Jupyter Notebook asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        "jupytext",
        "--to",
        "ipynb",
        str(md_file),
        "-o",
        str(ipynb_file),
    )
    await proc.wait()
    if proc.returncode == 0:
        print(f"Converted: {md_file} -> {ipynb_file}")
    else:
        print(f"Error converting {md_file}: Exit code {proc.returncode}")


async def process_file(
    session: aiohttp.ClientSession,
    github_file: ContentFile.ContentFile,
    output_base_dir: Path,
) -> None:
    """Downloads and converts a single file."""
    file_url: str = github_file.download_url
    # Extract relevant part of the path, excluding 'docs/source/'
    sub_path: Path = Path(github_file.path.replace("docs/source/", ""))
    folder: str = sub_path.parts[0]

    md_file: Path = output_base_dir / sub_path
    ipynb_file: Path = output_base_dir / folder / (Path(github_file.name).stem + ".ipynb")

    # Create intermediate directories for the .md file
    md_file.parent.mkdir(parents=True, exist_ok=True)

    await download_file_async(session, file_url, md_file)
    await convert_to_ipynb_async(md_file, ipynb_file)
    md_file.unlink()  # Remove the temporary .md file


async def main() -> None:
    """Downloads Markdown files from GitHub and converts them to Jupyter Notebooks."""
    repo_name: str = "pipefunc/pipefunc"  # Your GitHub repository name
    folders: list[str] = ["examples", "concepts"]
    output_base_dir: Path = Path("notebooks")
    access_token: str | None = os.environ.get(
        "GITHUB_TOKEN",
    )  # Highly recommended to set this environment variable

    # Create output directory structure
    output_base_dir.mkdir(exist_ok=True)
    for folder in folders:
        (output_base_dir / folder).mkdir(exist_ok=True)

    # GitHub API setup
    g = Github(access_token) if access_token else Github()  # Authenticated or unauthenticated
    # Warning: if unauthenticated, you might hit rate limits
    repo: Repository = g.get_repo(repo_name)

    # Download and convert files concurrently
    async with aiohttp.ClientSession() as session:
        tasks: list[asyncio.Task] = []
        for folder in folders:
            contents: list[ContentFile.ContentFile] = repo.get_contents(f"docs/source/{folder}")
            for file in contents:
                if file.name.endswith(".md"):
                    task = asyncio.create_task(process_file(session, file, output_base_dir))
                    tasks.append(task)

        await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
