"""Automatically generate release notes from GitHub issues and PRs."""

from __future__ import annotations

import datetime
import functools
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import git
from diskcache import Cache
from github import Github, Repository
from rich.console import Console

if TYPE_CHECKING:
    from collections.abc import Callable

console = Console()


def _print_step(message: str) -> None:
    """Print a step with a nice format."""
    console.print(f"[bold blue]\n➡️  {message}[/]\n")


def _print_substep(message: str) -> None:
    """Print a substep with a nice format."""
    console.print(f"  🔹 {message}")


def _print_info(message: str) -> None:
    """Print an info message with a nice format."""
    console.print(f"  ℹ️  {message}")  # noqa: RUF001


def _print_warning(message: str) -> None:
    """Print a warning message with a nice format."""
    console.print(f"[bold orange]  ⚠️  {message}[/]")


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


@_cached_github_call
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


def _categorize_pr_title(pr_title: str) -> str:  # noqa: PLR0911
    """Categorize a PR title based on prefixes."""
    if pr_title.startswith("DOC:"):
        return "📚 Documentation"
    if pr_title.startswith("ENH:"):
        return "✨ Enhancements"
    if pr_title.startswith("CI:"):
        return "🤖 CI"
    if pr_title.startswith("TST:"):
        return "🧪 Testing"
    if pr_title.startswith("MAINT:"):
        return "🧹 Maintenance"
    if pr_title.startswith("BUG:"):
        return "🐛 Bug Fixes"
    if pr_title.startswith("⬆️"):
        return "📦 Dependencies"
    if pr_title.startswith("[pre-commit.ci]"):
        return "🔄 Pre-commit"
    return "📝 Other"


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
def _get_all_closed_issues(gh_repo: Repository.Repository) -> list[Github.Issue.Issue]:
    """Get all closed issues (non-PR) from the repository."""
    _print_step("Getting all closed issues...")

    issues = []
    for issue in gh_repo.get_issues(state="closed"):
        if not issue.pull_request:  # Skip PRs, we only want issues
            _print_substep(f"Found issue {issue.title} (#{issue.number})")
            issues.append(issue)

    _print_info(f"Found {len(issues)} closed issues in total")
    return issues


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
        if "=>" in line:  # Skip renames (only because it is harder to parse)
            continue  # TODO: Fix this
        # Parse the line
        added, deleted, file_path = line.split("\t")
        try:
            name, extension = file_path.split(".", 1)
            if name == "":
                extension = "other"
        except ValueError:
            extension = "other"
        stats[extension]["added"] += int(added)
        stats[extension]["deleted"] += int(deleted)

    _print_info(f"Found changes in {len(stats)} file extensions")
    return stats


def _generate_release_notes(
    token: str,
    repo_path: str | None = None,
    remote_name: str = "origin",
) -> str:
    """Generate release notes in Markdown format."""
    repo, remote = _get_repo(repo_path, remote_name)
    gh = Github(token)
    gh_repo = _get_github_repo(gh, remote)

    # Get all tags sorted by date
    tags_with_dates = _get_tags_with_dates(repo)

    # Get all closed issues once
    all_closed_issues = _get_all_closed_issues(gh_repo)

    markdown = ""
    # Generate notes for each release
    for i, (tag, tag_date) in enumerate(tags_with_dates):
        prev_tag, prev_date = (
            tags_with_dates[i + 1] if i + 1 < len(tags_with_dates) else (None, None)
        )

        markdown += f"## Version {tag.name} ({tag_date.strftime('%Y-%m-%d')})\n\n"

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
            markdown += "### Closed Issues\n\n"
            for issue in issues:
                markdown += f"- {_get_first_line(issue.title)} (#{issue.number})\n"
            markdown += "\n"

        # Categorize and add commits
        commits_by_category: dict[str, list[str]] = defaultdict(list)
        for commit in commits:
            if not commit.message.startswith("Merge pull request"):
                category = _categorize_pr_title(commit.message)
                commits_by_category[category].append(_get_first_line(commit.message))

        # Add commits by category, only if there are commits in that category
        for category, messages in commits_by_category.items():
            if messages:
                markdown += f"### {category}\n\n"
                for msg in messages:
                    markdown += f"- {msg}\n"
                markdown += "\n"

        # Add file stats
        markdown += "### 📊 Stats\n\n"
        for extension, stats in file_stats.items():
            ext = f"`.{extension}`" if extension != "other" else "other"
            markdown += f"- {ext}: "
            markdown += f"+{stats['added']} lines, "
            markdown += f"-{stats['deleted']} lines\n"
        markdown += "\n"

    return markdown.rstrip() + "\n"  # Ensure single newline at end of file


if __name__ == "__main__":
    # Replace with your GitHub token
    with open("GITHUB_TOKEN") as f:  # noqa: PTH123
        github_token = f.read().strip()
    notes = _generate_release_notes(github_token)
    console.print(notes)
    with open("../RELEASE_NOTES.md", "w") as f:  # noqa: PTH123
        f.write(notes)
