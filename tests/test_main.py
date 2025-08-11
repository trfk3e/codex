import json
import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import GitHubAPIClient, RateLimitInfo


class MockResponse:
    def __init__(self, data: dict):
        self.status_code = 200
        self._lines = json.dumps(data).splitlines()

    def raise_for_status(self):
        pass

    def iter_lines(self, decode_unicode=True):
        for line in self._lines:
            yield line


def test_fetch_rate_limit_parses_stream():
    payload = {
        "resources": {
            "core": {"limit": 60, "remaining": 57, "reset": 123},
            "search": {"limit": 10, "remaining": 9, "reset": 456},
            "graphql": {"limit": 15, "remaining": 14, "reset": 789},
        }
    }

    session = SimpleNamespace(
        get=lambda url, stream=True, timeout=10: MockResponse(payload),
        headers={},
    )
    client = GitHubAPIClient("dummykey", session=session)
    status = client.fetch_rate_limit()

    assert isinstance(status.core, RateLimitInfo)
    assert status.core.limit == 60
    assert status.search.remaining == 9
    assert status.graphql.reset == 789
