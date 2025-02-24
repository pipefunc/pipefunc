"""Downloads all relevant documentation from PipeFunc's GitHub repository and converts it to Jupyter Notebooks.

Run this script like
```bash
uv run get-notebooks.py
```
or
```bash
uv run https://raw.githubusercontent.com/pipefunc/pipefunc/refs/heads/main/get-notebooks.py
```
"""

# /// script
# dependencies = [
# "jupytext",
# "PyGithub",
# "aiohttp",
# "rich",
# ]
# ///
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

import aiohttp
import rich
from github import ContentFile, Github, GithubException  # PyGithub
from rich.progress import Progress, TaskID
from rich.table import Table

if TYPE_CHECKING:
    from github.Repository import Repository


async def download_file_async(
    session: aiohttp.ClientSession,
    url: str,
    destination: Path,
    progress: Progress,
    task_id: TaskID,
) -> None:
    """Downloads a file asynchronously."""
    async with session.get(url) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        with destination.open("wb") as file:
            async for chunk in response.content.iter_chunked(8192):
                file.write(chunk)
                progress.update(task_id, advance=len(chunk), total=total_size)
        rich.print(f"⬇️ Downloaded: {url} -> {destination}")


async def convert_to_ipynb_async(md_file: Path, ipynb_file: Path) -> None:
    """Converts a Markdown file to a Jupyter Notebook asynchronously."""
    proc = await asyncio.create_subprocess_exec(
        "jupytext",
        "--to",
        "ipynb",
        str(md_file),
        "-o",
        str(ipynb_file),
        stdout=asyncio.subprocess.PIPE,  # Redirect stdout to a pipe
        stderr=asyncio.subprocess.PIPE,  # Redirect stderr to a pipe
    )
    await proc.wait()

    if proc.returncode == 0:
        rich.print(f"🔄 Converted: {md_file} -> {ipynb_file}")
    else:
        assert proc.stderr is not None
        stderr = await proc.stderr.read()
        rich.print(
            f"[bold red]❌ Error converting {md_file}: Exit code {proc.returncode}[/bold red]",
        )
        if stderr:
            rich.print(f"[bold red]Stderr: {stderr.decode()}[/bold red]")


async def download_and_convert_file(
    session: aiohttp.ClientSession,
    github_file: ContentFile.ContentFile,
    output_base_dir: Path,
    progress: Progress,
) -> None:
    """Downloads a file from GitHub and converts it to a Jupyter Notebook.

    Handles both Markdown files (for conversion) and .ipynb files (direct download).
    """
    file_url: str = github_file.download_url
    file_name: str = github_file.name

    if file_name.endswith(".md"):
        # Markdown files: Convert to .ipynb
        # Extract relevant part of the path, excluding 'docs/source/'
        sub_path: Path = Path(github_file.path.replace("docs/source/", ""))
        folder: str = sub_path.parts[0]

        md_file: Path = output_base_dir / sub_path
        ipynb_file: Path = output_base_dir / folder / (Path(file_name).stem + ".ipynb")

        # Create intermediate directories for the .md file
        md_file.parent.mkdir(parents=True, exist_ok=True)

        task_id = progress.add_task(f"Downloading {file_name}", start=False)
        await download_file_async(session, file_url, md_file, progress, task_id)
        progress.remove_task(task_id)
        await convert_to_ipynb_async(md_file, ipynb_file)
        md_file.unlink()  # Remove the temporary .md file
    elif file_name.endswith(".ipynb"):
        # .ipynb files: Download directly
        destination: Path = output_base_dir / file_name
        task_id = progress.add_task(f"Downloading {file_name}", start=False)
        await download_file_async(session, file_url, destination, progress, task_id)
        progress.remove_task(task_id)
    else:
        rich.print(f"⚠️ Skipping unsupported file type: {file_name}")


async def download_files_from_folder(
    session: aiohttp.ClientSession,
    repo: "Repository",
    folder: str,
    output_base_dir: Path,
    progress: Progress,
) -> None:
    """Downloads and converts files from a specific folder in the GitHub repository."""
    contents: list[ContentFile.ContentFile] = repo.get_contents(f"docs/source/{folder}")
    rich.print(f"🔍 Found {len(contents)} files in '{folder}' folder")
    for file in contents:
        await download_and_convert_file(session, file, output_base_dir, progress)


async def download_root_notebook(
    session: aiohttp.ClientSession,
    repo: "Repository",
    output_base_dir: Path,
    progress: Progress,
) -> None:
    """Downloads the example.ipynb notebook from the root of the repository."""
    try:
        file: ContentFile.ContentFile = repo.get_contents("example.ipynb")
        await download_and_convert_file(session, file, output_base_dir, progress)
    except GithubException as e:
        if e.status == 404:  # noqa: PLR2004
            rich.print("⚠️  Could not find example.ipynb in the root of the repository.")
        else:
            rich.print(f"[bold red]❌ Error downloading example.ipynb: {e}[/bold red]")


async def main() -> None:
    """Downloads Markdown files from GitHub and converts them to Jupyter Notebooks."""
    rich.print("🚀 Starting conversion process...")
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
    rich.print(f"📂 Created output directory structure: {output_base_dir}")

    # GitHub API setup
    g = Github(access_token) if access_token else Github()
    repo: Repository = g.get_repo(repo_name)
    rich.print(f"🐙 Connected to GitHub repository: {repo_name}")

    # Download and convert files concurrently
    async with aiohttp.ClientSession() as session:
        with Progress() as progress:
            # Download example.ipynb from the root
            await download_root_notebook(session, repo, output_base_dir, progress)

            tasks: list[asyncio.Task] = [
                asyncio.create_task(
                    download_files_from_folder(session, repo, folder, output_base_dir, progress),
                )
                for folder in folders
            ]

            await asyncio.gather(*tasks)

    rich.print("🎉 Finished conversion process!")

    # Create a table of generated files
    table = Table(title="Generated Notebooks")
    table.add_column("File", justify="left", style="cyan", no_wrap=True)
    table.add_column("Path", justify="left", style="magenta")

    for root, _, files in os.walk(output_base_dir):
        for file in files:
            if file.endswith(".ipynb"):
                file_path = Path(root) / file
                table.add_row(file, str(file_path.relative_to(output_base_dir)))

    rich.print(table)


if __name__ == "__main__":
    asyncio.run(main())
