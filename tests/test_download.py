"""download.py 集成测试：respx mock Civitai 元数据 + 假 aria2 落盘，覆盖编排、进度与失败分支。"""

import asyncio
import json
import os
import struct
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from fastapi import HTTPException
from pydantic import ValidationError

import comfyui_anima_api.loras.aria2c as aria2c
import comfyui_anima_api.loras.download as dl
from comfyui_anima_api.config import CIVITAI_CONFIG
from comfyui_anima_api.schemas import DownloadParams

CIVITAI = "https://civitai.test"


def lora_blob(max_block: int = 5) -> bytes:
    """构造最小 safetensors（2B 布局内，块号 <= 27）。"""
    header = json.dumps({f"lora_transformer.blocks.{max_block}.attn": {}}).encode()
    return struct.pack("<Q", len(header)) + header


def patch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(CIVITAI_CONFIG, "loras_download_dir", str(tmp_path))
    monkeypatch.setattr(CIVITAI_CONFIG, "base_url", CIVITAI)
    monkeypatch.setattr(aria2c, "is_available", lambda: True)


def mock_meta(**overrides: Any) -> respx.Route:
    meta = {"baseModel": "Anima", "files": [{"type": "Model", "hashes": {"SHA256": None}}]}
    meta.update(overrides)
    return respx.get(f"{CIVITAI}/api/v1/model-versions/42").mock(return_value=httpx.Response(200, json=meta))


def fake_aria2_writing(blob: bytes) -> Callable[..., Awaitable[None]]:
    async def run(
        url: str,
        header_args: list[str],
        dest_dir: str,
        out_name: str,
        *,
        timeout: float,
        on_progress: aria2c.ProgressCallback | None = None,
        cancel: asyncio.Event | None = None,
        proxy: str | None = None,
    ) -> None:
        if on_progress is not None:
            progress = aria2c._parse_aria2_progress("[#z 1MiB/10MiB(10%) CN:8 DL:2MiB ETA:4s]")
            if progress:
                await on_progress(progress)
        with open(os.path.join(dest_dir, out_name), "wb") as f:
            f.write(blob)

    return run


def call(p: DownloadParams) -> dict:
    return asyncio.run(dl.run(p))


P: dict[str, Any] = dict(version_id=42, filename="x.safetensors")


class TestHappyPath:
    def test_download_saves_and_verifies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """正常路径：precheck -> aria2 落盘 -> 校验 -> 改名 -> 响应，无 .part 残留。"""
        patch_env(monkeypatch, tmp_path)
        blob = lora_blob()
        monkeypatch.setattr(aria2c, "run", fake_aria2_writing(blob))
        with respx.mock:
            mock_meta()
            result = call(DownloadParams(**P))
        assert result == {"filename": "x.safetensors", "version_id": 42}
        assert (tmp_path / "x.safetensors").read_bytes() == blob
        assert not (tmp_path / "x.safetensors.part").exists()  # 已改名，不残留分片


class TestDownloadFailures:
    def test_sha_mismatch_discards_and_502(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_env(monkeypatch, tmp_path)
        monkeypatch.setattr(aria2c, "run", fake_aria2_writing(lora_blob()))
        with respx.mock:
            mock_meta(files=[{"type": "Model", "hashes": {"SHA256": "deadbeef"}}])
            with pytest.raises(HTTPException) as ei:
                call(DownloadParams(**P))
        assert ei.value.status_code == 502
        assert not (tmp_path / "x.safetensors.part").exists()  # 校验失败即删除
        assert not (tmp_path / "x.safetensors").exists()

    def test_aria2_failure_keeps_partial_and_hints_api_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_env(monkeypatch, tmp_path)
        monkeypatch.setattr(CIVITAI_CONFIG, "api_key", None)

        async def failing_aria2(
            url: str,
            header_args: list[str],
            dest_dir: str,
            out_name: str,
            *,
            timeout: float,
            on_progress: aria2c.ProgressCallback | None = None,
            cancel: asyncio.Event | None = None,
            proxy: str | None = None,
        ) -> None:
            with open(os.path.join(dest_dir, out_name), "wb") as f:
                f.write(b"partial")
            raise RuntimeError("aria2c exit code 1")

        monkeypatch.setattr(aria2c, "run", failing_aria2)
        with respx.mock:
            mock_meta()
            with pytest.raises(HTTPException) as ei:
                call(DownloadParams(**P))
        assert ei.value.status_code == 502
        assert (tmp_path / "x.safetensors.part").exists()  # 断点续传：残片保留
        assert "CIVITAI_API_KEY" in ei.value.detail

    def test_aria2c_missing_is_503_without_network(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(CIVITAI_CONFIG, "loras_download_dir", str(tmp_path))
        monkeypatch.setattr(aria2c, "is_available", lambda: False)
        with pytest.raises(HTTPException) as ei:
            call(DownloadParams(**P))
        assert ei.value.status_code == 503
        assert "not installed" in ei.value.detail

    def test_path_traversal_filename_rejected_at_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 文件名校验统一进了 schema（原 downloader/route 的重复检查已删）：
        # 路径穿越在 DownloadParams 构造期即 ValidationError（HTTP 侧由 test_api 的
        # test_download_rejects_path_traversal_without_network 覆盖 400）
        patch_env(monkeypatch, tmp_path)
        with pytest.raises(ValidationError, match="plain '<name>\\.safetensors'"):
            DownloadParams(version_id=42, filename="../evil.safetensors")


class TestAria2ProgressParsing:
    def test_typical_summary_line(self) -> None:
        p = aria2c._parse_aria2_progress("[#8b1f3c 45MiB/1.9GiB(2%) CN:8 DL:5.2MiB ETA:5m54s]")
        assert p is not None
        assert p["bytes"] == 45 * 1024**2
        assert p["total"] == int(1.9 * 1024**3)
        assert p["speed_bps"] == int(5.2 * 1024**2)
        assert p["eta_s"] == 5 * 60 + 54
        assert p["percent"] == pytest.approx(2.3, abs=0.1)

    def test_eta_variants(self) -> None:
        p = aria2c._parse_aria2_progress("[#a 0B/1KiB(0%) CN:1 DL:0B ETA:n/a]")
        assert p is not None and p["eta_s"] is None and p["bytes"] == 0
        p2 = aria2c._parse_aria2_progress("[#b 1GiB/2GiB(50%) CN:2 DL:1GiB ETA:1h2m3s]")
        assert p2 is not None and p2["eta_s"] == 3723
        # ETA 段缺失也能解析
        p3 = aria2c._parse_aria2_progress("[#c 10MiB/20MiB(50%) CN:2 DL:1MiB]")
        assert p3 is not None and p3["eta_s"] is None

    def test_non_progress_lines_return_none(self) -> None:
        for line in ["(OK): download completed.", "Download Results:", "", "[#x SEED 1MiB]", " gid    status"]:
            assert aria2c._parse_aria2_progress(line) is None


class TestAria2HeaderArgs:
    def test_formats_header_pairs(self) -> None:
        assert aria2c.build_headers({"User-Agent": "ua", "Authorization": "Bearer x"}) == [
            "--header",
            "User-Agent: ua",
            "--header",
            "Authorization: Bearer x",
        ]


class TestDownloadProgress:
    def test_on_progress_gets_initial_estimate_then_aria2_updates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """进度回调两段式：元数据尺寸 + 已有残片先给一帧，aria2 行解析后给真实帧。"""
        patch_env(monkeypatch, tmp_path)
        seen = []

        async def cb(d: dict[str, Any]) -> None:
            seen.append(d)

        # 假文件补齐到元数据声明的 10240B（header 后补零不影响 safetensors 头解析），
        # 否则会撞上"下载体与元数据大小对账"防线
        blob = lora_blob().ljust(10240, b"\0")
        monkeypatch.setattr(aria2c, "run", fake_aria2_writing(blob))
        with respx.mock:
            mock_meta(files=[{"type": "Model", "hashes": {"SHA256": None}, "sizeKB": 10.0}])
            asyncio.run(dl.run(DownloadParams(**P), on_progress=cb))
        assert seen[0]["total"] == 10240 and seen[0]["bytes"] == 0  # sizeKB=10 -> 10240B，无残片
        assert seen[1]["bytes"] == 1024**2 and seen[1]["speed_bps"] == 2 * 1024**2  # aria2 帧
        assert len(seen) == 2

    def test_initial_progress_accounts_existing_part_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """续传场景：已有 .part 残片计入首帧 bytes/percent。"""
        patch_env(monkeypatch, tmp_path)
        (tmp_path / "x.safetensors.part").write_bytes(b"x" * 5120)
        seen = []

        async def cb(d: dict[str, Any]) -> None:
            seen.append(d)

        blob = lora_blob().ljust(10240, b"\0")
        monkeypatch.setattr(aria2c, "run", fake_aria2_writing(blob))
        with respx.mock:
            mock_meta(files=[{"type": "Model", "hashes": {"SHA256": None}, "sizeKB": 10.0}])
            asyncio.run(dl.run(DownloadParams(**P), on_progress=cb))
        assert seen[0]["bytes"] == 5120 and seen[0]["percent"] == 50.0


class _FakeAria2Stdout:
    def __init__(self, proc: "_FakeAria2Proc") -> None:
        self._proc = proc

    def __aiter__(self) -> "_FakeAria2Stdout":
        return self

    async def __anext__(self) -> bytes:
        if self._proc.returncode is not None:
            raise StopAsyncIteration
        await asyncio.sleep(0.005)
        return b"[#f1 1MiB/10MiB(10%) CN:8 DL:1MiB ETA:9s]"


class _FakeAria2Proc:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self._closed = asyncio.Event()
        self.stdout = _FakeAria2Stdout(self)

    async def wait(self) -> int:
        await self._closed.wait()
        return self.returncode or 0

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = 9
        self._closed.set()


class TestRunAria2Cancel:
    def test_cancel_kills_process_and_raises_cancelled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        proc = _FakeAria2Proc()

        async def fake_exec(*args: Any, **kwargs: Any) -> _FakeAria2Proc:
            return proc

        monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
        cancel = asyncio.Event()
        seen = []

        async def on_progress(d: dict[str, Any]) -> None:
            seen.append(d)

        async def scenario() -> None:
            task = asyncio.create_task(
                aria2c.run("http://x/", [], str(tmp_path), "o.part", timeout=10, on_progress=on_progress, cancel=cancel)
            )
            for _ in range(400):
                if seen:
                    break
                await asyncio.sleep(0.005)
            cancel.set()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(scenario())
        assert seen and seen[0]["bytes"] == 1024**2 and seen[0]["eta_s"] == 9.0
        assert proc.returncode == 9  # 取消即杀进程
