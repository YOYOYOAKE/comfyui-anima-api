"""
路由层测试：TestClient + respx mock ComfyUI HTTP API。
同步校验类用裸 TestClient（任务未创建即可断言 400/503）；任务流类必须用 flow fixture
（with TestClient：lifespan 生效 + portal 事件循环贯穿多个请求，后台任务才能跨请求推进），
并把 ComfyUI 的 ws 进度通道 patch 成空操作，等待路径确定性走 /history 轮询。
"""

import asyncio
import io
import json
import time
import zipfile
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from comfyui_anima_api.app import app
from comfyui_anima_api.comfyui import comfyui_progress
from comfyui_anima_api.config import CIVITAI_CONFIG, COMFYUI_CONFIG
from comfyui_anima_api.loras import registry
from comfyui_anima_api.tasks import get_store, reconcile

COMFY = "http://comfy-test:8188"

# 真 PNG 文件头的 b64：decode_b64_image 按 magic bytes 预检，裸 b"x" 之类会被 400
PNG_B64 = "iVBORw0KGgo="

client = TestClient(app)


@pytest.fixture(autouse=True)
def patch_comfyui_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(COMFYUI_CONFIG, "base_url", COMFY)


def patch_loras_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(CIVITAI_CONFIG, "loras_download_dir", str(tmp_path))


@pytest.fixture()
def flow(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    async def _noop() -> None:
        pass

    monkeypatch.setattr(comfyui_progress, "start", _noop)
    monkeypatch.setattr(comfyui_progress, "stop", _noop)
    with TestClient(app) as c:
        yield c


def history_entry(pid: str, filename: str = "001_.png", subfolder: str = "anima/t1/001") -> dict:
    return {
        pid: {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {"9": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}},
        }
    }


def wait_status(c: TestClient, task_id: str, *statuses: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    task: dict = {}
    while time.monotonic() < deadline:
        task = c.get(f"/tasks/{task_id}").json()
        if task["status"] in statuses:
            return task
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} stuck in {task.get('status')!r}, wanted {statuses}")


def stream_events(c: TestClient, task_id: str, last_event_id: int = 0) -> list[tuple[int, str, dict]]:
    """收一次完整 SSE 流（终态后流自关）：返回 [(seq, event, data)]。"""
    with c.stream("GET", f"/tasks/{task_id}/events", headers={"Last-Event-ID": str(last_event_id)}) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events: list[tuple[int, str, dict]] = []
        seq, etype = None, None
        for line in resp.iter_lines():
            if line.startswith("id: "):
                seq = int(line[4:])
            elif line.startswith("event: "):
                etype = line[7:]
            elif line.startswith("data: ") and seq is not None and etype is not None:
                events.append((seq, etype, json.loads(line[6:])))
                seq, etype = None, None
    return events


class TestHealth:
    def test_ok_when_comfyui_healthy(self) -> None:
        with respx.mock:
            respx.get(f"{COMFY}/system_stats").mock(return_value=httpx.Response(200, json={}))
            r = client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "comfyui": "ok"}

    def test_503_when_comfyui_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(COMFYUI_CONFIG, "base_url", "http://127.0.0.1:1")
        r = client.get("/health")
        assert r.status_code == 503
        assert "error" in r.json()


class TestGenerateValidation:
    """同步校验：任务创建前即 400，无需 mock 网络。"""

    def test_empty_positive_is_400(self) -> None:
        r = client.post("/anima/generate", json={"prompt": {"positive": ""}})
        assert r.status_code == 400
        assert "error" in r.json()

    def test_whitespace_positive_is_400(self) -> None:
        # 空白串不算空 prompt：点名拒绝，区别于"没填"
        r = client.post("/anima/generate", json={"prompt": {"positive": "   "}})
        assert r.status_code == 400
        assert "whitespace" in r.json()["error"]

    def test_generate_rejects_upscale_group(self) -> None:
        # 超分走 POST /upscale：generate 不认识 upscale 参数
        r = client.post("/anima/generate", json={"prompt": {"positive": "x"}, "upscale": {"enable": True}})
        assert r.status_code == 400
        assert "upscale" in r.json()["error"]

    def test_invalid_base64_is_400_immediately(self) -> None:
        # b64 预检在路由层：坏图与坏参数一样即时拒绝，不进任务
        r = client.post("/anima/generate", json={"prompt": {"positive": "x"}, "base": {"image": "!!!not-b64"}})
        assert r.status_code == 400
        assert "base.image" in r.json()["error"]

    def test_invalid_count_is_400(self) -> None:
        r = client.post("/anima/generate", json={"prompt": {"positive": "x"}, "count": 0})
        assert r.status_code == 400

    def test_batch_with_explicit_seed_is_400(self, flow: TestClient) -> None:
        # 批量不支持指定种子：count>1 且显式 seed 直接拒，全用随机
        r = flow.post("/anima/generate", json={"prompt": {"positive": "x"}, "seed": 7, "count": 3})
        assert r.status_code == 400
        assert "seed" in r.json()["error"]
        assert "count>1" in r.json()["error"]
        # 显式 null 等价未设置：不触发本条拒绝（走正常任务创建；history 空=任务停在轮询即可）
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-null"}))
            respx.get(f"{COMFY}/history/pid-null").mock(return_value=httpx.Response(200, json={}))
            r2 = flow.post("/anima/generate", json={"prompt": {"positive": "x"}, "seed": None, "count": 3})
        assert r2.status_code == 202

    def test_reference_fields_without_image_is_400(self) -> None:
        r = client.post("/anima/generate", json={"prompt": {"positive": "x"}, "reference": {"mode": "depth"}})
        assert r.status_code == 400
        assert "reference.mode" in r.json()["error"]

    def test_reference_mask_with_non_inpaint_is_400(self) -> None:
        r = client.post(
            "/anima/generate",
            json={"prompt": {"positive": "x"}, "reference": {"image": PNG_B64, "mode": "mix", "mask": PNG_B64}},
        )
        assert r.status_code == 400
        assert "reference.mask" in r.json()["error"]

    def test_invalid_enum_is_400(self) -> None:
        r = client.post("/anima/generate", json={"prompt": {"positive": "x"}, "size": "huge"})
        assert r.status_code == 400


class TestGenerateFlow:
    def _mock_comfyui(self, pid: str = "pid-1") -> respx.Route:
        respx.get(f"{COMFY}/history/{pid}").mock(return_value=httpx.Response(200, json=history_entry(pid)))
        respx.get(f"{COMFY}/view").mock(return_value=httpx.Response(200, content=b"\x89PNG-fake"))
        return respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": pid}))

    def test_happy_path_202_then_done_then_zip(self, flow: TestClient) -> None:
        with respx.mock:
            self._mock_comfyui("pid-1")
            r = flow.post("/anima/generate", json={"prompt": {"positive": "test"}, "seed": 1})
            assert r.status_code == 202
            task_id = r.json()["task_id"]
            task = wait_status(flow, task_id, "done", "error")
            assert task["status"] == "done"
            assert task["images_done"] == 1
            assert task["images"][0]["filename"] == "001_.png"
            assert task["images"][0]["url"] == f"/tasks/{task_id}/images/1"
            assert task["prompt_id"] == "pid-1"
            z = flow.get(f"/tasks/{task_id}/images.zip")
        assert z.status_code == 200
        assert z.headers["content-type"] == "application/zip"
        assert z.headers["x-images-count"] == "1"
        assert z.content[:2] == b"PK"
        names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
        assert names == ["001_001_.png"]  # ComfyUI 取图健康则无 _INCOMPLETE.txt marker

    def test_events_stream_full_lifecycle(self, flow: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
        # 终态清理后，完成任务的回放只剩终态事件——完整事件序列必须在任务在跑时订阅
        # 才能看到：history 前 0.3s 返回空（任务停在轮询），SSE 先回放 queued/running，
        # 再实时收 image_done/done 关流；完成后重连回放仅剩 done
        monkeypatch.setattr(COMFYUI_CONFIG, "poll_interval", 0.01)
        started = time.monotonic()

        def gated_history(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={} if time.monotonic() - started < 0.3 else history_entry("pid-2"))

        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-2"}))
            respx.get(f"{COMFY}/history/pid-2").mock(side_effect=gated_history)
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            time.sleep(0.05)  # 让 running 先进回放
            events = stream_events(flow, task_id)
            after_done = stream_events(flow, task_id)
        assert [e[1] for e in events] == ["status", "status", "image_done", "done"]
        assert events[0][2]["status"] == "queued"
        assert events[1][2]["status"] == "running"
        assert events[2][2]["image"] == 1 and events[2][2]["images_total"] == 1
        assert events[2][2]["url"] == f"/tasks/{task_id}/images/1"
        assert events[3][2]["status"] == "done" and events[3][2]["images"] == 1
        # 终态清理：完成后的重连回放只剩终态事件
        assert [e[1] for e in after_done] == ["done"]

    def test_last_event_id_past_terminal_closes_immediately(self, flow: TestClient) -> None:
        # EventSource 在服务端正常关流后仍会自动重连一次：Last-Event-ID 已越过终态
        # 事件时回放为空且任务已终态，流必须立即收尾（而非挂着发心跳）
        with respx.mock:
            self._mock_comfyui("pid-3")
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            wait_status(flow, task_id, "done")
            full = stream_events(flow, task_id)
        assert [e[1] for e in full] == ["done"]
        partial = stream_events(flow, task_id, last_event_id=full[-1][0])
        assert partial == []

    def test_count_submits_per_image_with_random_seeds(self, flow: TestClient) -> None:
        with respx.mock:
            prompt_route = self._mock_comfyui("pid-4")
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}, "count": 3}).json()["task_id"]
            task = wait_status(flow, task_id, "done", "error")
        assert task["status"] == "done" and task["images_done"] == 3
        assert prompt_route.call_count == 3
        bodies = [json.loads(c.request.content) for c in prompt_route.calls]
        seeds = [b["prompt"]["anima_sampler"]["inputs"]["seed"] for b in bodies]
        prefixes = [b["prompt"]["anima_save"]["inputs"]["filename_prefix"] for b in bodies]
        # 批量全随机：三张各自独立种子（互不相同、无 +i 规律），产物按任务子目录 + 序号命名
        assert len(set(seeds)) == 3
        assert all(0 <= s < 2**64 for s in seeds)
        assert prefixes[0].endswith("/001") and prefixes[1].endswith("/002") and prefixes[2].endswith("/003")

    def test_single_image_seed_passthrough(self, flow: TestClient) -> None:
        with respx.mock:
            prompt_route = self._mock_comfyui("pid-12")
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}, "seed": 7}).json()["task_id"]
            wait_status(flow, task_id, "done", "error")
        bodies = [json.loads(c.request.content) for c in prompt_route.calls]
        assert [b["prompt"]["anima_sampler"]["inputs"]["seed"] for b in bodies] == [7]

    def test_workflow_rejected_becomes_error_task(self, flow: TestClient) -> None:
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"node_errors": {"7": "bad"}}))
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            task = wait_status(flow, task_id, "error")
            events = stream_events(flow, task_id)
        assert "workflow rejected" in task["error"]
        error_event = events[-1]
        assert error_event[1] == "error"
        assert error_event[2]["http_status"] == 400
        assert "workflow rejected" in error_event[2]["message"]

    def test_comfyui_unreachable_becomes_502_error(self, flow: TestClient) -> None:
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(side_effect=httpx.ConnectError("boom"))
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            task = wait_status(flow, task_id, "error")
            events = stream_events(flow, task_id)
        assert events[-1][2]["http_status"] == 502
        assert "cannot reach ComfyUI" in task["error"]

    def test_comfyui_non_json_response_is_502_error(self, flow: TestClient) -> None:
        # ComfyUI 返回反代错误页等非 JSON 体 -> 502，而非裸 500
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(502, text="<html>bad gateway</html>"))
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            task = wait_status(flow, task_id, "error")
        assert "non-JSON" in task["error"]


class TestUpscaleFlow:
    def _mock_comfyui(self, pid: str, content: bytes) -> respx.Route:
        respx.post(f"{COMFY}/upload/image").mock(return_value=httpx.Response(200, json={"name": "anima_upscale_x.png"}))
        respx.get(f"{COMFY}/history/{pid}").mock(
            return_value=httpx.Response(
                200,
                json=history_entry(pid, filename="up_.png", subfolder="upscale/t9/001"),
            )
        )
        respx.get(f"{COMFY}/view").mock(return_value=httpx.Response(200, content=content))
        return respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": pid}))

    def test_happy_path(self, flow: TestClient) -> None:
        with respx.mock:
            self._mock_comfyui("pid-5", b"\x89PNG-up")
            r = flow.post("/upscale", json={"image": PNG_B64})
            assert r.status_code == 202
            task_id = r.json()["task_id"]
            task = wait_status(flow, task_id, "done", "error")
            z = flow.get(f"/tasks/{task_id}/images.zip")
        assert task["status"] == "done" and task["images_done"] == 1
        assert z.content[:2] == b"PK"

    def test_image_required(self) -> None:
        r = client.post("/upscale", json={})
        assert r.status_code == 400
        assert "image" in r.json()["error"]

    def test_unknown_field_rejected(self) -> None:
        r = client.post("/upscale", json={"image": PNG_B64, "enable": True})
        assert r.status_code == 400
        assert "enable" in r.json()["error"]

    def test_explicit_null_timeout_is_default(self, flow: TestClient) -> None:
        # null 一律等价未设置
        with respx.mock:
            self._mock_comfyui("pid-6", b"\x89PNG-up2")
            r = flow.post("/upscale", json={"image": PNG_B64, "timeout": None})
        assert r.status_code == 202


class TestCancel:
    def test_cancel_running_generation(self, flow: TestClient) -> None:
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-9"}))
            # history 永远为空：任务停在轮询等待，取消才有可观察窗口
            respx.get(f"{COMFY}/history/pid-9").mock(return_value=httpx.Response(200, json={}))
            respx.get(f"{COMFY}/queue").mock(
                return_value=httpx.Response(200, json={"queue_running": [], "queue_pending": []})
            )
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            wait_status(flow, task_id, "running")
            r = flow.post(f"/tasks/{task_id}/cancel")
            assert r.status_code == 200
            assert r.json()["status"] == "cancelling"
            task = wait_status(flow, task_id, "cancelled")
        assert task["status"] == "cancelled"
        events = stream_events(flow, task_id)
        assert events[-1][1] == "cancelled"

    def test_cancel_terminal_task_is_idempotent(self, flow: TestClient) -> None:
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-10"}))
            respx.get(f"{COMFY}/history/pid-10").mock(return_value=httpx.Response(200, json=history_entry("pid-10")))
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            wait_status(flow, task_id, "done")
            r = flow.post(f"/tasks/{task_id}/cancel")
        assert r.json() == {"task_id": task_id, "status": "done"}


class TestComfyuiHelpers:
    """comfyui.cancel 纯函数单测：/queue 条目解析（真机 ComfyUI 0.43 实测格式）。"""

    def test_queue_prompt_ids_all_entry_shapes(self) -> None:
        from comfyui_anima_api.comfyui.cancel import _queue_prompt_ids

        # 现行格式：[number, prompt_id, prompt, ...]——prompt_id 在第二位（首位是序号）
        current = [[4, "7c5211b9-7839-4213-97ed-0494abc13ddb", {"1": {}}], [5, "00000000-dead-beef", {}]]
        assert _queue_prompt_ids(current) == {"7c5211b9-7839-4213-97ed-0494abc13ddb", "00000000-dead-beef"}
        # 老格式：prompt_id 在首位
        assert _queue_prompt_ids([["pid-old", {}]]) == {"pid-old"}
        # 对象格式
        assert _queue_prompt_ids([{"prompt_id": "pid-obj"}]) == {"pid-obj"}
        # 空值/怪值不炸、不产出
        assert _queue_prompt_ids(None) == set()
        assert _queue_prompt_ids([]) == set()
        assert _queue_prompt_ids([[4, 5, {}], [""]]) == set()


class TestTasksRoutes:
    def test_unknown_task_404_everywhere(self, flow: TestClient) -> None:
        assert flow.get("/tasks/nope").status_code == 404
        assert flow.get("/tasks/nope/events").status_code == 404
        assert flow.get("/tasks/nope/images.zip").status_code == 404
        assert flow.get("/tasks/nope/images/1").status_code == 404
        assert flow.post("/tasks/nope/cancel").status_code == 404

    def test_zip_rejected_until_task_terminal(self, flow: TestClient) -> None:
        # 未终态一律 409：半套图片没有消费价值，调用方不必处理"包不完整"的判断
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-11"}))
            respx.get(f"{COMFY}/history/pid-11").mock(return_value=httpx.Response(200, json={}))  # 任务停在轮询
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            r = flow.get(f"/tasks/{task_id}/images.zip")
        assert r.status_code == 409
        assert "terminal state" in r.json()["error"]

    def test_zip_terminal_without_images_is_404(self, flow: TestClient) -> None:
        # 终态（error）但一张图都没产出：404 no images
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-e"}))
            entry = history_entry("pid-e")["pid-e"]
            entry["status"] = {"status_str": "error", "completed": False, "messages": []}
            respx.get(f"{COMFY}/history/pid-e").mock(return_value=httpx.Response(200, json={"pid-e": entry}))
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            wait_status(flow, task_id, "error")
            r = flow.get(f"/tasks/{task_id}/images.zip")
        assert r.status_code == 404
        assert "no images" in r.json()["error"]

    def test_zip_midstream_failure_marks_incomplete(self, flow: TestClient) -> None:
        # 流式打包的代价：响应开始后无法改口状态码——中途取图失败以 _INCOMPLETE.txt 收尾
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-z"}))
            respx.get(f"{COMFY}/history/pid-z").mock(return_value=httpx.Response(200, json=history_entry("pid-z")))
            respx.get(f"{COMFY}/view").mock(side_effect=httpx.ConnectError("boom"))
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}}).json()["task_id"]
            wait_status(flow, task_id, "done", "error")
            z = flow.get(f"/tasks/{task_id}/images.zip")
        assert z.status_code == 200
        names = zipfile.ZipFile(io.BytesIO(z.content)).namelist()
        assert names == ["_INCOMPLETE.txt"]

    def test_single_image_midrun_available_with_zip_still_409(
        self, flow: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # 单图端点无终态门禁:第 1 张落地即 200 可取;此刻 zip 仍 409、
        # 未产出的 idx 404(detail 带进度提示)、idx=0 过不了参数校验
        monkeypatch.setattr(COMFYUI_CONFIG, "poll_interval", 0.01)
        prompt_calls = 0

        def prompt_side(request: httpx.Request) -> httpx.Response:
            nonlocal prompt_calls
            prompt_calls += 1
            return httpx.Response(200, json={"prompt_id": f"pid-m{prompt_calls}"})

        def history_side(request: httpx.Request) -> httpx.Response:
            pid = request.url.path.rsplit("/", 1)[-1]
            return httpx.Response(200, json=history_entry(pid) if pid == "pid-m1" else {})

        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(side_effect=prompt_side)
            respx.get(url__startswith=f"{COMFY}/history/").mock(side_effect=history_side)
            view = respx.get(f"{COMFY}/view").mock(return_value=httpx.Response(200, content=b"\x89PNG-single"))
            task_id = flow.post("/anima/generate", json={"prompt": {"positive": "x"}, "count": 2}).json()["task_id"]
            # 等 count=2 批次的第 1 张落地;第 2 张 history 恒空,任务停在 running
            snap: dict = {}
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                snap = flow.get(f"/tasks/{task_id}").json()
                if snap["images_done"] == 1:
                    break
                time.sleep(0.05)
            assert snap["images_done"] == 1 and snap["status"] == "running"
            assert snap["images"][0]["url"] == f"/tasks/{task_id}/images/1"

            ok = flow.get(f"/tasks/{task_id}/images/1")
            assert ok.status_code == 200
            assert ok.headers["content-type"] == "image/png"
            assert ok.headers["content-disposition"] == 'attachment; filename="001_.png"'
            assert ok.content == b"\x89PNG-single"
            assert view.call_count == 1

            # 对照:同一时刻 zip 仍被终态门禁拦下;未产出的 idx 404 带进度;idx=0 走全局校验处理器 400
            assert flow.get(f"/tasks/{task_id}/images.zip").status_code == 409
            not_ready = flow.get(f"/tasks/{task_id}/images/2")
            assert not_ready.status_code == 404
            assert "no image 2" in not_ready.json()["error"] and "images_done=1" in not_ready.json()["error"]
            bad_idx = flow.get(f"/tasks/{task_id}/images/0")
            assert bad_idx.status_code == 400 and "invalid request" in bad_idx.json()["error"]


class TestParamsSanitized:
    """入库参数脱敏：b64 图片字段替换为占位符——快照只回显占位符，库行不随原图膨胀。"""

    def test_generate_params_strip_base64(self, flow: TestClient) -> None:
        with respx.mock:
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-s1"}))
            respx.get(f"{COMFY}/history/pid-s1").mock(return_value=httpx.Response(200, json={}))
            task_id = flow.post(
                "/anima/generate",
                json={"prompt": {"positive": "x"}, "base": {"image": PNG_B64, "mask": PNG_B64}},
            ).json()["task_id"]
        params = flow.get(f"/tasks/{task_id}").json()["params"]
        assert params["base"]["image"] == "<base64 omitted: 12 chars>"
        assert params["base"]["mask"] == "<base64 omitted: 12 chars>"
        assert params["prompt"]["positive"] == "x"  # 非 b64 字段原样保留

    def test_upscale_params_strip_base64(self, flow: TestClient) -> None:
        with respx.mock:
            respx.post(f"{COMFY}/upload/image").mock(return_value=httpx.Response(200, json={"name": "u.png"}))
            respx.post(f"{COMFY}/prompt").mock(return_value=httpx.Response(200, json={"prompt_id": "pid-s2"}))
            respx.get(f"{COMFY}/history/pid-s2").mock(return_value=httpx.Response(200, json={}))
            task_id = flow.post("/upscale", json={"image": PNG_B64}).json()["task_id"]
        params = flow.get(f"/tasks/{task_id}").json()["params"]
        assert params["image"] == "<base64 omitted: 12 chars>"


class TestReconcile:
    """重启对账：最后一个 prompt 已完成的任务收尾 done，残批的 done 事件附 truncated。"""

    def test_completed_batch_marked_done_with_truncation(self) -> None:
        store = get_store()
        asyncio.run(store.create_task("rec1", "generate", {"prompt": {"positive": "x"}}, images_total=3))
        asyncio.run(store.update_task("rec1", status="running", started_at=time.time(), prompt_id="pid-rec"))
        # 重启前已完成 1/3 张；重启后 ComfyUI 把最后一个 prompt 跑完了（又出 1 张）
        asyncio.run(
            store.add_image(
                "rec1", {"filename": "000_.png", "subfolder": "anima/rec1/000", "type": "output"}, "pid-rec"
            )
        )
        with respx.mock:
            respx.get(f"{COMFY}/history/pid-rec").mock(
                return_value=httpx.Response(
                    200, json=history_entry("pid-rec", filename="001_.png", subfolder="anima/rec1/001")
                )
            )
            asyncio.run(reconcile())
        task = asyncio.run(store.get_task("rec1"))
        assert task is not None
        assert task["status"] == "done" and task["images_done"] == 2
        events = asyncio.run(store.events_after("rec1", 0))
        assert events[-1]["type"] == "done" and events[-1]["data"]["truncated"] is True

    def test_completed_full_batch_has_no_truncation_flag(self) -> None:
        store = get_store()
        asyncio.run(store.create_task("rec2", "generate", {"prompt": {"positive": "x"}}, images_total=1))
        asyncio.run(store.update_task("rec2", status="running", started_at=time.time(), prompt_id="pid-rec2"))
        with respx.mock:
            respx.get(f"{COMFY}/history/pid-rec2").mock(
                return_value=httpx.Response(200, json=history_entry("pid-rec2"))
            )
            asyncio.run(reconcile())
        task = asyncio.run(store.get_task("rec2"))
        assert task is not None
        assert task["status"] == "done" and task["images_done"] == 1
        events = asyncio.run(store.events_after("rec2", 0))
        assert events[-1]["type"] == "done" and "truncated" not in events[-1]["data"]


class TestLoraRoutes:
    def test_list_empty_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        r = client.get("/lora/list")
        assert r.status_code == 200
        assert r.json() == {"counts": 0, "loras": []}

    def test_list_with_registered_lora(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        patch_loras_paths(monkeypatch, tmp_path)
        (tmp_path / "a.safetensors").write_bytes(b"x")
        registry.update_registry("a.safetensors", 42, "trigger a")
        r = client.get("/lora/list")
        body = r.json()
        assert body["counts"] == 1
        assert body["loras"][0] == {"filename": "a.safetensors", "version_id": 42, "trigger": "trigger a"}

    def test_download_rejects_path_traversal_without_network(self) -> None:
        r = client.post("/lora/download", json={"version_id": 1, "filename": "../evil.safetensors", "trigger": "t"})
        assert r.status_code == 400

    def test_download_rejects_api_key_in_body(self) -> None:
        # 鉴权凭据只认服务端 CIVITAI_API_KEY，请求体传入即未知字段
        r = client.post(
            "/lora/download",
            json={"version_id": 1, "filename": "a.safetensors", "trigger": "t", "api_key": "sk-from-caller"},
        )
        assert r.status_code == 400
        assert "api_key" in r.json()["error"]

    def test_download_rejects_non_json_body(self) -> None:
        r = client.post("/lora/download", content="garbage", headers={"Content-Type": "application/json"})
        assert r.status_code == 400

    def test_download_503_when_aria2c_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import comfyui_anima_api.routes.loras as loras_route

        monkeypatch.setattr(loras_route, "aria2c_available", lambda: False)
        r = client.post("/lora/download", json={"version_id": 1, "filename": "a.safetensors", "trigger": "t"})
        assert r.status_code == 503
        assert "not installed" in r.json()["error"]
