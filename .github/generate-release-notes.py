"""Automatically generate release notes from GitHub issues and PRs.

Run `uv run .github/generate-release-notes.py` to generate the release notes.
Or
```bash
GITHUB_TOKEN=$(gh auth token) uv run .github/generate-release-notes.py
```
to use the GitHub CLI to authenticate.
"""
# /// script
# dependencies = [
#   "rich",
#   "PyGithub",
#   "diskcache",
#   "GitPython",
# ]
# ///

from __future__ import annotations

import datetime
import functools
import os
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import git
from diskcache import Cache
from github import Github, Repository
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable

REPO_ROOT = Path(__file__).parent.parent
console = Console()
REPO_URL = "https://github.com/pipefunc/pipefunc"


def _print_step(message: str) -> None:
    """Print a step with a nice format."""
    console.print(f"[bold blue]\n➡️  {message}[/]\n")


def _print_substep(message: str) -> None:
    """Print a substep with a nice format."""
    console.print(f"  🔹 {message}")


def _print_info(message: str) -> None:
    """Print an info message with a nice format."""
    console.print(f"  ℹ️  {message}")  # noqa: RUF001


def _print_error(message: str) -> None:
    """Print an error message with a nice format."""
    console.print(f"[bold red]  ❌  {message}[/]")


def _get_repo(
    repo_path: str | None = None,
    remote_name: str = "origin",
) -> tuple[git.Repo, git.Remote]:
    """Get the git repository and the remote."""
    _print_step("Getting git repository and remote...")
    repo = git.Repo(repo_path, search_parent_directories=True)
    _print_info(f"Using repository at {repo.working_dir}")
    remote = repo.remote(remote_name)
    _print_info(f"Using remote {remote.name}")
    return repo, remote


def _cached_github_call(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to cache GitHub API calls."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202, ANN002, ANN003
        def _serialize_for_cache(obj):  # noqa: ANN202, ANN001
            """Serialize objects to be used in cache keys."""
            if isinstance(obj, git.TagReference):
                return obj.name  # Use tag name
            if isinstance(obj, datetime.datetime):
                return obj.strftime("%Y-%m-%d")  # Use formatted date string
            if isinstance(obj, Repository.Repository):
                return obj.full_name  # use repo full name
            if isinstance(obj, git.Remote):
                return obj.name
            return obj

        # Serialize arguments and keyword arguments for cache key
        serialized_args = tuple(_serialize_for_cache(a) for a in args if not isinstance(a, Github))
        serialized_kwargs = tuple(
            (k, _serialize_for_cache(v)) for k, v in kwargs.items() if not isinstance(v, Github)
        )

        cache_key = (func.__name__, serialized_args, serialized_kwargs)
        _print_substep(f"Cache key: {cache_key}")

        with Cache("~/.cache/github_cache") as cache:
            if cache_key in cache:
                _print_info("Using cached result")
                return cache[cache_key]

            _print_info("Fetching from GitHub API")
            result = func(*args, **kwargs)
            cache[cache_key] = result
            return result

    return wrapper


def _get_github_repo(gh: Github, remote: git.Remote) -> Repository.Repository:
    """Get the GitHub repository."""
    _print_step("Getting GitHub repository...")
    # Assuming the remote URL is a GitHub URL
    remote_url = remote.url
    if remote_url.startswith("git@github.com:"):
        repo_name = remote_url.replace("git@github.com:", "").replace(".git", "")
    elif remote_url.startswith("https://github.com/"):
        repo_name = remote_url.replace("https://github.com/", "").replace(".git", "")
    else:
        _print_error(f"Remote URL {remote_url} is not a GitHub URL")
        msg = f"Remote URL {remote_url} is not a GitHub URL"
        raise ValueError(msg)

    _print_info(f"Using repository {repo_name}")
    return gh.get_repo(repo_name)


def _categorize_pr_title(pr_title: str) -> tuple[int, str]:
    """Categorize a PR title based on prefixes."""
    mapping = {
        "DOC:": "📚 Documentation",
        "ENH:": "✨ Enhancements",
        "CI:": "🤖 CI",
        "TST:": "🧪 Testing",
        "MAINT:": "🧹 Maintenance",
        "BUG:": "🐛 Bug Fixes",
        "FIX:": "🐛 Bug Fixes",  # Used this a couple of times, but now stick to "BUG:"
        "⬆️": "📦 Dependencies",
        "[pre-commit.ci]": "🔄 Pre-commit",
    }
    for prefix, category in mapping.items():
        if pr_title.startswith(prefix):
            n = len(prefix) + 1 if prefix.endswith(":") else 0
            return n, category
    return 0, "📝 Other"


def _get_tags_with_dates(repo: git.Repo) -> list[tuple[git.TagReference, datetime.datetime]]:
    """Get all tags with their dates, sorted by date (newest first)."""
    _print_step("Getting tags with dates...")
    tags_with_dates = []
    for tag in repo.tags:
        # Get the commit date for the tag
        commit_date = datetime.datetime.fromtimestamp(
            tag.commit.committed_date,
            tz=datetime.timezone.utc,
        )
        tags_with_dates.append((tag, commit_date))

    # Sort by date, newest first
    return sorted(tags_with_dates, key=lambda x: x[1], reverse=True)


@_cached_github_call
def _get_issue(gh_repo: Repository.Repository, number: int) -> Github.Issue.Issue | None:
    """Get a single issue by number, with caching."""
    try:
        return gh_repo.get_issue(number)
    except Exception:  # Some numbers might not exist due to deleted issues  # noqa: BLE001
        return None


@_cached_github_call
def _get_latest_issue_number(gh_repo: Repository.Repository) -> int:
    """Get the number of the latest issue."""
    # Get the latest issue (including PRs) to get the highest number
    latest = gh_repo.get_issues(state="all", sort="created", direction="desc")[0]
    return latest.number


def _get_all_closed_issues(gh_repo: Repository.Repository) -> list[Github.Issue.Issue]:
    """Get all closed issues (non-PR) from the repository."""
    _print_step("Getting all closed issues...")

    latest_number = _get_latest_issue_number(gh_repo)
    _print_info(f"Latest issue number is {latest_number}")

    # Fetch all issues in parallel
    with ThreadPoolExecutor() as executor:
        all_issues = list(
            executor.map(_get_issue, [gh_repo] * latest_number, range(1, latest_number + 1)),
        )

    # Filter the issues
    filtered_issues = []
    for issue in all_issues:
        if issue is not None and issue.state == "closed" and not issue.pull_request:
            _print_substep(f"Found issue {issue.title} (#{issue.number})")
            filtered_issues.append(issue)

    _print_info(f"Found {len(filtered_issues)} closed issues in total")
    return filtered_issues


def _filter_issues_by_timeframe(
    issues: list[Github.Issue.Issue],
    start_date: datetime.datetime | None,
    end_date: datetime.datetime,
) -> list[Github.Issue.Issue]:
    """Filter issues that were closed between start_date and end_date."""
    filtered = [
        issue
        for issue in issues
        if issue.closed_at <= end_date and (start_date is None or issue.closed_at > start_date)
    ]
    _print_info(f"Found {len(filtered)} issues closed in this time window")
    return filtered


def _get_pr_nr(line: str) -> str | None:
    pattern = r"\(#(\d+)\)$"  # The regex pattern

    match = re.search(pattern, line)

    if match:
        return match.group(1)
    return None


def _get_first_line(message: str) -> str:
    """Get the first line of a multi-line message."""
    return message.split("\n")[0].strip()


def _get_file_stats(
    repo: git.Repo,
    start_commit: str,
    end_commit: str,
) -> dict[str, dict[str, int]]:
    """Get statistics about lines added, deleted, and changed per file extension."""
    _print_step("Getting file stats...")
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    # Get the diff between the start and end commits
    # Use --find-renames to handle renames
    diff_index = repo.git.diff("--numstat", "--find-renames", start_commit, end_commit)

    for line in diff_index.split("\n"):
        if not line:
            continue
        # Parse the line
        added, deleted, file_path = line.split("\t")
        try:
            name, extension = file_path.rsplit(".", 1)
            if name == "":
                extension = "other"
            if " =>" in extension:
                extension = extension.split(" =>")[0]
        except ValueError:
            extension = "other"
        stats[extension]["added"] += int(added)
        stats[extension]["deleted"] += int(deleted)

    _print_info(f"Found changes in {len(stats)} file extensions")
    return stats


def _generate_release_notes(  # noqa: PLR0912, PLR0915
    token: str | None = None,
    repo_path: str | None = None,
    remote_name: str = "origin",
) -> tuple[str, str]:
    """Generate release notes in Markdown format."""
    repo, remote = _get_repo(repo_path, remote_name)
    gh = Github(token)
    gh_repo = _get_github_repo(gh, remote)

    # Get all tags sorted by date
    tags_with_dates = _get_tags_with_dates(repo)

    # Get all closed issues once
    all_closed_issues = _get_all_closed_issues(gh_repo)

    lines: list[str] = []
    lines.append("# Changelog\n\n")
    lines.append(
        "These release notes are automatically generated from commits and GitHub issues and PRs.\n",
    )
    lines.append(
        "If it is out of date, please run `GITHUB_TOKEN=$(gh auth token) uv run .github/generate-release-notes.py`.\n\n",
    )
    # Generate notes for each release
    for i, (tag, tag_date) in enumerate(tags_with_dates):
        prev_tag, prev_date = (
            tags_with_dates[i + 1] if i + 1 < len(tags_with_dates) else (None, None)
        )

        lines.append(f"## {tag.name} ({tag_date.strftime('%Y-%m-%d')})\n\n")

        # Get commits
        if prev_tag is None:
            commits = list(repo.iter_commits(tag.name))
            start_commit = tag.name
            end_commit = f"{tag.name}~1"
        else:
            commits = list(repo.iter_commits(f"{prev_tag.name}..{tag.name}"))
            start_commit = prev_tag.name
            end_commit = tag.name

        # Get file stats
        file_stats = _get_file_stats(repo, start_commit, end_commit)

        # Filter issues for this time window
        issues = _filter_issues_by_timeframe(all_closed_issues, prev_date, tag_date)

        # Add closed issues
        if issues:
            lines.append("### Closed Issues\n\n")
            for issue in sorted(issues, key=lambda x: x.number, reverse=True):
                url = f"[#{issue.number}]({REPO_URL}/issues/{issue.number})"
                lines.append(f"- {_get_first_line(issue.title)} ({url})\n")
            lines.append("\n")

        # Categorize and add commits
        commits_by_category: dict[str, list[str]] = defaultdict(list)
        for commit in commits:
            if not commit.message.startswith("Merge pull request"):
                n_skip, category = _categorize_pr_title(commit.message)
                line = _get_first_line(commit.message)
                if (pr_nr := _get_pr_nr(line)) is not None:
                    link = f"[#{pr_nr}]({REPO_URL}/pull/{pr_nr})"
                    line = line.replace(f"(#{pr_nr})", f"({link})")
                else:
                    ref = commit.hexsha[:7]
                    link = f"[{ref}]({REPO_URL}/commit/{ref})"
                    line = f"{line} ({link})"
                commits_by_category[category].append(line[n_skip:])

        # Add commits by category, only if there are commits in that category
        for category, messages in commits_by_category.items():
            if messages:
                lines.append(f"### {category}\n\n")
                for msg in messages:
                    lines.append(f"- {msg}\n")  # noqa: PERF401
                lines.append("\n")

        # Add file stats
        lines.append("### 📊 Stats\n\n")
        for extension, stats in file_stats.items():
            if extension == "other":
                continue
            _add_stats_lines(lines, f"`.{extension}`", stats)
        if (stats := file_stats.get("other")) is not None:  # type: ignore[assignment]
            _add_stats_lines(lines, "`other`", stats)
        lines.append("\n")
    last_tag = tags_with_dates[0][0].name
    return "".join(lines).rstrip() + "\n", last_tag


def _add_stats_lines(lines: list[str], ext: str, stats: dict[str, int]) -> None:
    lines.append(f"- {ext}: ")
    lines.append(f"+{stats['added']} lines, ")
    lines.append(f"-{stats['deleted']} lines\n")


if __name__ == "__main__":
    _print_step("Generating release notes...")
    token_file = REPO_ROOT / ".github" / "GITHUB_TOKEN"
    if token_file.exists():
        with token_file.open() as f:
            _print_info("Using GitHub token from file")
            github_token = f.read().strip()
    elif "GITHUB_TOKEN" in os.environ:
        _print_info("Using GitHub token from environment")
        github_token = os.environ["GITHUB_TOKEN"]
    else:
        _print_error("No GitHub token found")
        github_token = None  # type: ignore[assignment]
    notes, last_tag = _generate_release_notes(github_token)
    console.print(notes)
    with open(REPO_ROOT / "CHANGELOG.md", "w") as f:  # noqa: PTH123
        f.write(notes)

    _print_info("Run the following command to commit the changes:")
    _print_info(
        f"git checkout -b update-changelog-{last_tag} && git add CHANGELOG.md && git commit -m 'DOC: Update `CHANGELOG.md` until {last_tag}' && git push",
    )
