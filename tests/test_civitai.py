"""civitai.py 单测：元数据获取、Anima 准入、primary 文件选择与请求头。"""

import asyncio
import hashlib
from typing import Any

import httpx
import pytest
import respx
from fastapi import HTTPException

from comfyui_anima_api.config import CIVITAI_CONFIG
from comfyui_anima_api.loras import civitai

CIVITAI = "https://civitai.test"


def patch_civitai(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(CIVITAI_CONFIG, "base_url", CIVITAI)


def mock_meta(**overrides: Any) -> respx.Route:
    meta: dict[str, Any] = {"baseModel": "Anima", "files": [{"type": "Model", "hashes": {"SHA256": None}}]}
    meta.update(overrides)
    return respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(return_value=httpx.Response(200, json=meta))


class TestFetchVersion:
    def test_returns_primary_file_metadata(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """多文件 version：SHA/大小对准 primary——/api/download 实际服务的文件。"""
        patch_civitai(monkeypatch)
        blob_sha = hashlib.sha256(b"blob").hexdigest()
        with respx.mock:
            mock_meta(
                files=[
                    {"type": "Model", "hashes": {"SHA256": "deadbeef"}},
                    {"type": "Model", "primary": True, "hashes": {"SHA256": blob_sha}, "sizeKB": 10.0},
                ]
            )
            info = asyncio.run(civitai.fetch_version(42))
        assert info == civitai.VersionInfo(
            version_id=42, base_model="Anima", sha256=blob_sha, size_bytes=10 * 1024
        )

    def test_unknown_version_is_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        with respx.mock:
            respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(return_value=httpx.Response(404, json={}))
            with pytest.raises(HTTPException) as ei:
                asyncio.run(civitai.fetch_version(42))
        assert ei.value.status_code == 404

    def test_non_anima_basemodel_is_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        with respx.mock:
            mock_meta(baseModel="Illustrious XL")
            with pytest.raises(HTTPException) as ei:
                asyncio.run(civitai.fetch_version(42))
        assert ei.value.status_code == 400
        assert "only Anima" in ei.value.detail

    def test_auth_rejected_hints_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        monkeypatch.setattr(CIVITAI_CONFIG, "api_key", None)
        with respx.mock:
            respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(return_value=httpx.Response(401, json={}))
            with pytest.raises(HTTPException) as ei:
                asyncio.run(civitai.fetch_version(42))
        assert ei.value.status_code == 502
        assert "CIVITAI_API_KEY" in ei.value.detail

    def test_auth_rejected_hints_invalid_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        monkeypatch.setattr(CIVITAI_CONFIG, "api_key", "sk-bad")
        with respx.mock:
            respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(return_value=httpx.Response(403, json={}))
            with pytest.raises(HTTPException) as ei:
                asyncio.run(civitai.fetch_version(42))
        assert ei.value.status_code == 502
        assert "may be invalid" in ei.value.detail

    def test_metadata_request_carries_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        monkeypatch.setattr(CIVITAI_CONFIG, "api_key", "sk-test")
        with respx.mock:
            route = respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(
                return_value=httpx.Response(200, json={"baseModel": "Anima", "files": []})
            )
            asyncio.run(civitai.fetch_version(42))
        assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"

    def test_non_json_metadata_is_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        with respx.mock:
            respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(
                return_value=httpx.Response(200, text="not json")
            )
            with pytest.raises(HTTPException) as ei:
                asyncio.run(civitai.fetch_version(42))
        assert ei.value.status_code == 502
        assert "non-JSON" in ei.value.detail

    def test_non_object_metadata_is_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        with respx.mock:
            respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(return_value=httpx.Response(200, json=[]))
            with pytest.raises(HTTPException) as ei:
                asyncio.run(civitai.fetch_version(42))
        assert ei.value.status_code == 502
        assert "non-object" in ei.value.detail

    def test_network_error_is_502(self, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_civitai(monkeypatch)
        with respx.mock:
            respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(side_effect=httpx.ConnectError("boom"))
            with pytest.raises(HTTPException) as ei:
                asyncio.run(civitai.fetch_version(42))
        assert ei.value.status_code == 502
        assert "cannot reach civitai" in ei.value.detail


class TestBuildRequestHeaders:
    def test_headers_include_ua_extra_and_bearer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CIVITAI_CONFIG, "ua", "test-ua")
        monkeypatch.setattr(CIVITAI_CONFIG, "extra_headers", {"X-Test": "1"})
        monkeypatch.setattr(CIVITAI_CONFIG, "api_key", "sk")
        assert civitai.build_request_headers() == {
            "User-Agent": "test-ua",
            "X-Test": "1",
            "Authorization": "Bearer sk",
        }

    def test_headers_omit_authorization_without_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CIVITAI_CONFIG, "ua", "test-ua")
        monkeypatch.setattr(CIVITAI_CONFIG, "extra_headers", {})
        monkeypatch.setattr(CIVITAI_CONFIG, "api_key", None)
        assert civitai.build_request_headers() == {"User-Agent": "test-ua"}


class TestBuildDownloadUrl:
    def test_uses_configured_base_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CIVITAI_CONFIG, "base_url", CIVITAI)
        assert civitai.build_download_url(42) == f"{CIVITAI}/api/download/models/42"
