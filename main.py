"""GitHub API key monitor.

This script queries the GitHub API to obtain detailed rate limit
information for each API key provided. It processes the API response
line by line to ensure every chunk of data is handled correctly and
provides the most informative output per key.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import requests

GITHUB_API_URL = "https://api.github.com/rate_limit"


@dataclass
class RateLimitInfo:
    """Represents rate limit information for a GitHub resource."""

    limit: int
    remaining: int
    reset: int

    @classmethod
    def from_dict(cls, data: dict) -> "RateLimitInfo":
        return cls(limit=data.get("limit", 0),
                   remaining=data.get("remaining", 0),
                   reset=data.get("reset", 0))


@dataclass
class APIKeyStatus:
    """Aggregated status information for a single API key."""

    key_suffix: str
    core: RateLimitInfo
    search: RateLimitInfo
    graphql: RateLimitInfo

    def pretty(self) -> str:
        return (
            f"Key …{self.key_suffix}: \n"
            f"  Core    -> {self.core.remaining}/{self.core.limit}, reset at {self.core.reset}\n"
            f"  Search  -> {self.search.remaining}/{self.search.limit}, reset at {self.search.reset}\n"
            f"  GraphQL -> {self.graphql.remaining}/{self.graphql.limit}, reset at {self.graphql.reset}"
        )


class GitHubAPIClient:
    """Client for communicating with the GitHub API using a specific key."""

    def __init__(self, api_key: str, session: Optional[requests.Session] = None) -> None:
        self.api_key = api_key
        self.session = session or requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    def fetch_rate_limit(self) -> APIKeyStatus:
        """Fetch rate limit information by processing the HTTP stream line by line."""

        logging.debug("Requesting rate limit info from GitHub API")
        response = self.session.get(GITHUB_API_URL, stream=True, timeout=10)
        response.raise_for_status()

        lines: list[str] = []
        for line in response.iter_lines(decode_unicode=True):
            if line:
                logging.debug("API line received: %s", line)
                lines.append(line)

        data = json.loads("".join(lines))
        resources = data.get("resources", {})

        core = RateLimitInfo.from_dict(resources.get("core", {}))
        search = RateLimitInfo.from_dict(resources.get("search", {}))
        graphql = RateLimitInfo.from_dict(resources.get("graphql", {}))

        key_suffix = self.api_key[-4:] if self.api_key else ""
        return APIKeyStatus(key_suffix, core, search, graphql)


def read_keys(path: str) -> Iterable[str]:
    """Yield API keys from the given file, one per line."""

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key:
                yield key


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="GitHub API key monitor")
    parser.add_argument("file", help="Path to a file containing API keys, one per line")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    for key in read_keys(args.file):
        try:
            client = GitHubAPIClient(key)
            status = client.fetch_rate_limit()
            print(status.pretty())
        except requests.HTTPError as exc:
            logging.error("HTTP error for key …%s: %s", key[-4:], exc)
        except requests.RequestException as exc:
            logging.error("Request failed for key …%s: %s", key[-4:], exc)
        except json.JSONDecodeError as exc:
            logging.error("Failed to parse response for key …%s: %s", key[-4:], exc)


if __name__ == "__main__":
    main()
