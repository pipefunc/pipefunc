"""Automatically generate release notes from GitHub issues and PRs."""

from __future__ import annotations

import datetime
import functools
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import git
from diskcache import Cache
from github import Github, Repository, UnknownObjectException
from packaging import version
from rich import print
from rich.console import Console
from rich.progress import track

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


def _get_tags_with_dates(repo: git.Repo) -> list[tuple[git.Tag, datetime.datetime]]:
    """Get a list of tags with their creation dates."""
    _print_step("Getting tags with dates...")
    tags_with_dates = []
    for tag in repo.tags:
        try:
            tag_date = datetime.datetime.fromtimestamp(
                tag.object.tagged_date,
                tz=datetime.timezone.utc,
            )
        except AttributeError:
            tag_date = datetime.datetime.fromtimestamp(
                tag.object.committed_date,
                tz=datetime.timezone.utc,
            )
        _print_substep(f"Found tag {tag.name} with date {tag_date.strftime('%Y-%m-%d')}")
        tags_with_dates.append((tag, tag_date))
    return sorted(tags_with_dates, key=lambda x: x[1], reverse=True)


def _get_commits_between(
    repo: git.Repo,
    start_tag: git.Tag | None,
    end_tag: git.Tag,
) -> list[git.Commit]:
    """Get a list of commits between two tags."""
    if start_tag is None:
        _print_step(f"Getting commits up to {end_tag.name}")
        commits = list(repo.iter_commits(end_tag.name))
    else:
        _print_step(
            f"Getting commits between {start_tag.name} and {end_tag.name if end_tag else 'HEAD'}",
        )
        # Use start_tag as the end point for the first iteration
        end_commit = start_tag.commit if end_tag is None else end_tag.commit
        commits = list(repo.iter_commits(f"{end_commit}...{start_tag.commit}"))
    _print_info(f"Found {len(commits)} commits")
    return commits


def _extract_pr_number(commit_message: str) -> int | None:
    """Extract the PR number from a commit message."""
    match = re.search(r"\(#(\d+)\)", commit_message)
    return int(match.group(1)) if match else None


@_cached_github_call
def _get_closed_issues_between_tags(
    gh_repo: Repository.Repository,
    start_tag: git.Tag | None,
    end_tag: git.Tag,
    start_date: datetime.datetime | None,
    end_date: datetime.datetime,
) -> list[Github.Issue.Issue]:
    """Get a list of closed issues between two tags."""
    if start_tag is None:
        _print_step(
            f"Getting closed issues before {end_tag.name} ({end_date.strftime('%Y-%m-%d')})",
        )
    else:
        _print_step(
            f"Getting closed issues between {start_tag.name} ({start_date.strftime('%Y-%m-%d')}) "
            f"and {end_tag.name} ({end_date.strftime('%Y-%m-%d')})",
        )
    issues = []
    kwargs: dict[str, Any] = {"state": "closed"}
    if start_date is not None:
        kwargs["since"] = start_date

    for issue in gh_repo.get_issues(**kwargs):
        if issue.closed_at > end_date:
            continue
        if start_date is not None and issue.closed_at < start_date:
            break
        if issue.pull_request:
            continue  # Skip PRs, we only want issues
        _print_substep(f"Found issue {issue.title} (#{issue.number})")
        issues.append(issue)
    _print_info(f"Found {len(issues)} closed issues")
    return issues


@_cached_github_call
def _get_pr_details(
    gh_repo: Repository.Repository,
    pr_number: int,
) -> tuple[str, str, datetime.datetime] | None:
    """Get the title, body, and merge date of a PR."""
    _print_step(f"Getting details for PR #{pr_number}...")
    try:
        pr = gh_repo.get_pull(pr_number)
    except UnknownObjectException:
        _print_warning(f"Could not find PR #{pr_number}")
        return None
    else:
        _print_info(f"Found PR: {pr.title}")
        return pr.title, pr.body, pr.merged_at


def _categorize_pr_title(pr_title: str) -> str:  # noqa: PLR0911
    """Categorize a PR title based on prefixes."""
    if pr_title.startswith("DOC:"):
        return "Documentation"
    if pr_title.startswith("ENH:"):
        return "Enhancements"
    if pr_title.startswith("CI:"):
        return "CI"
    if pr_title.startswith("TST:"):
        return "Testing"
    if pr_title.startswith("MAINT:"):
        return "Maintenance"
    if pr_title.startswith("BUG:"):
        return "Bug Fixes"
    if pr_title.startswith("⬆️"):
        return "Dependencies"
    if pr_title.startswith("[pre-commit.ci]"):
        return "Pre-commit"
    return "Other"


def _get_merged_prs_between_tags(
    repo: git.Repo,
    start_tag: git.Tag | None,
    end_tag: git.Tag,
) -> list[int]:
    """Get a list of PR numbers that were merged between two tags."""
    _print_step(
        f"Getting PRs merged between {start_tag.name if start_tag else 'HEAD'} and {end_tag.name}",
    )
    if start_tag is None:
        commits = list(repo.iter_commits(end_tag.name))
    else:
        commits = list(repo.iter_commits(f"{start_tag.name}...{end_tag.name}"))

    pr_numbers: list[int] = []
    for commit in commits:
        if commit.message.startswith("Merge pull request"):
            pr_number = _extract_pr_number(commit.message)
            if pr_number:
                pr_numbers.append(pr_number)
    _print_info(f"Found {len(pr_numbers)} merged PRs")
    return pr_numbers


def _generate_release_notes(
    token: str,
    repo_path: str | None = None,
    remote_name: str = "origin",
) -> str:
    """Generate release notes in Markdown format."""
    repo, remote = _get_repo(repo_path, remote_name)
    gh = Github(token)
    gh_repo = _get_github_repo(gh, remote)

    tags_with_dates = _get_tags_with_dates(repo)

    markdown = ""
    for i, (tag, tag_date) in enumerate(tags_with_dates):
        prev_tag = tags_with_dates[i + 1][0] if i + 1 < len(tags_with_dates) else None

        # Use Git log to get merged PRs
        merged_pr_numbers = _get_merged_prs_between_tags(repo, prev_tag, tag)

        tag_version = version.parse(tag.name)
        markdown += f"## Version {tag_version} ({tag_date.strftime('%Y-%m-%d')})\n\n"

        # Get closed issues between the two dates
        prev_tag_date = tags_with_dates[i + 1][1] if i + 1 < len(tags_with_dates) else None
        closed_issues = _get_closed_issues_between_tags(
            gh_repo,
            prev_tag,
            tag,
            prev_tag_date,
            tag_date,
        )
        if closed_issues:
            markdown += "### Closed Issues\n\n"
            for issue in closed_issues:
                markdown += f"- {issue.title} (#{issue.number})\n"
            markdown += "\n"

        # Categorize and add merged PRs to markdown
        prs_by_category: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        for pr_number in merged_pr_numbers:
            pr_details = _get_pr_details(gh_repo, pr_number)
            if pr_details is not None:
                pr_title, pr_body, _ = pr_details
                category = _categorize_pr_title(pr_title)
                prs_by_category[category].append((pr_title, pr_body, pr_number))

        for category, prs in prs_by_category.items():
            markdown += f"### {category}\n\n"
            for pr_title, _, pr_number in prs:
                markdown += f"- {pr_title} (#{pr_number})\n"
            markdown += "\n"

        # Get commits between the two tags, excluding merge commits for PRs
        commits = _get_commits_between(repo, prev_tag, tag)
        commits_by_category: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for commit in track(commits, description="Processing commits..."):
            if not commit.message.startswith("Merge pull request"):
                category = _categorize_pr_title(commit.message)
                commits_by_category[category].append((commit.message, ""))

        # Add commits to markdown
        for category, commits_in_category in commits_by_category.items():
            markdown += f"### {category}\n\n"
            for commit_message, pr_link in commits_in_category:
                markdown += f"- {commit_message} {pr_link}\n"
            if commits_in_category:
                markdown += "\n"

    return markdown


if __name__ == "__main__":
    # Replace with your GitHub token
    with open("GITHUB_TOKEN") as f:  # noqa: PTH123
        github_token = f.read().strip()
    notes = _generate_release_notes(github_token)
    print(notes)
    with open("../RELEASE_NOTES.md", "w") as f:  # noqa: PTH123
        f.write(notes)
