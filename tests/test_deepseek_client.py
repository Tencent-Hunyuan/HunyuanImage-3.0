# -*- coding: utf-8 -*-
"""Regression tests for PE.deepseek.DeepSeekClient.

The Tencent Cloud SDK is an optional install step (see README.md), so these
tests stub it out and drive DeepSeekClient against a fake ``common_client``
that records every request it receives.
"""
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeCommonClient(object):
    """Stand-in for tencentcloud's CommonClient that counts API requests."""

    def __init__(self, *args, **kwargs):
        self.calls = []
        self.fail_on = set()

    def _call_and_deserialize(self, action, params, resp_cls):
        self.calls.append(action)
        if len(self.calls) in self.fail_on:
            raise RuntimeError("simulated transient API error")
        resp = resp_cls()
        resp._deserialize(
            {"Choices": [{"Message": {
                "Content": "an enhanced prompt",
                "ReasoningContent": "the reasoning",
            }}]}
        )
        return resp


def _sdk_stubs():
    exception = type("TencentCloudSDKException", (Exception,), {})
    return {
        "tencentcloud": {},
        "tencentcloud.common": {},
        "tencentcloud.common.common_client": {"CommonClient": FakeCommonClient},
        "tencentcloud.common.credential": {"Credential": lambda *a, **k: object()},
        "tencentcloud.common.exception": {},
        "tencentcloud.common.exception.tencent_cloud_sdk_exception": {
            "TencentCloudSDKException": exception,
        },
        "tencentcloud.common.profile": {},
        "tencentcloud.common.profile.client_profile": {
            "ClientProfile": type("ClientProfile", (object,), {}),
        },
        "tencentcloud.common.profile.http_profile": {
            "HttpProfile": type("HttpProfile", (object,), {}),
        },
    }


@pytest.fixture
def client(monkeypatch):
    for name, attrs in _sdk_stubs().items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["tencentcloud.common"].credential = \
        sys.modules["tencentcloud.common.credential"]
    monkeypatch.delitem(sys.modules, "PE.deepseek", raising=False)

    import PE.deepseek as deepseek

    monkeypatch.setattr(deepseek.time, "sleep", lambda _seconds: None)
    return deepseek.DeepSeekClient("key-id", "key-secret")


def test_recaption_issues_exactly_one_request(client):
    """One recaption must cost one (billed) ChatCompletions request, not two."""
    content, reason = client.run_single_recaption("system", "a dog on the grass")

    assert content == "an enhanced prompt"
    assert reason == "the reasoning"
    assert client.common_client.calls == ["ChatCompletions"]


def test_transient_error_is_absorbed_by_the_retry_loop(client):
    """A transient failure must be retried, never surfaced to the caller."""
    client.common_client.fail_on = {1}

    content, _ = client.run_single_recaption("system", "a dog on the grass")

    assert content == "an enhanced prompt"
    assert client.common_client.calls == ["ChatCompletions"] * 2
