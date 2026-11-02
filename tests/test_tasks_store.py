"""tasks 持久层与事件总线单测：CRUD、事件回放（Last-Event-ID 语义）、订阅无缝衔接与终态回收。"""

import asyncio
import sqlite3
from collections.abc import Coroutine
from pathlib import Path
from typing import Any

import pytest

from comfyui_anima_api.config import TASKS_CONFIG
from comfyui_anima_api.tasks import bus as bus_mod
from comfyui_anima_api.tasks import get_store
from comfyui_anima_api.tasks import store as store_mod
from comfyui_anima_api.tasks.bus import _TaskBus
from comfyui_anima_api.tasks.sse import _format_event
from comfyui_anima_api.tasks.store import _TaskStore


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


class TestTaskStore:
    def test_task_lifecycle_and_events(self, tmp_path: Path) -> None:
        s = _TaskStore(str(tmp_path))
        run(s.create_task("t1", "generate", {"seed": 1}, images_total=3))
        task = run(s.get_task("t1"))
        assert task is not None
        assert task["status"] == "queued" and task["params"] == {"seed": 1} and task["images_total"] == 3
        assert run(s.get_task("missing")) is None

        run(s.update_task("t1", status="running", started_at=1.0, prompt_id="pid"))
        task = run(s.get_task("t1"))
        assert task is not None
        assert task["status"] == "running" and task["prompt_id"] == "pid"

        run(s.insert_event("t1", 1, "status", {"status": "running"}))
        run(s.insert_event("t1", 2, "progress", {"step": 1}))
        assert run(s.max_event_seq()) == 2
        assert [e["type"] for e in run(s.events_after("t1", 0))] == ["status", "progress"]
        assert [e["type"] for e in run(s.events_after("t1", 1))] == ["progress"]  # 严格大于 after_seq

        assert run(s.add_image("t1", {"filename": "a.png"}, "pid")) == 1
        assert run(s.add_image("t1", {"filename": "b.png", "subfolder": "s", "type": "output"}, "pid")) == 2
        done = run(s.get_task("t1"))
        assert done is not None and done["images_done"] == 2
        assert [i["filename"] for i in run(s.images("t1"))] == ["a.png", "b.png"]

        assert [t["id"] for t in run(s.unfinished_tasks())] == ["t1"]
        run(s.update_task("t1", status="done", finished_at=2.0))
        assert run(s.unfinished_tasks()) == []

    def test_update_rejects_unknown_column(self, tmp_path: Path) -> None:
        s = _TaskStore(str(tmp_path))
        run(s.create_task("t", "generate", {}))
        with pytest.raises(ValueError, match="unknown task columns"):
            run(s.update_task("t", hijacked="x"))

    def test_old_events_schema_migrates_on_open(self, tmp_path: Path) -> None:
        # 旧布局库文件（seq 单列主键）打开即迁移为 (task_id, seq) 复合主键，行原值保留
        conn = sqlite3.connect(str(tmp_path / "tasks.db"))
        conn.executescript(
            "CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,"
            " type TEXT NOT NULL, data TEXT NOT NULL, ts REAL NOT NULL);"
            "INSERT INTO events (task_id, type, data, ts) VALUES ('t', 'done', '{}', 1.0);"
        )
        conn.commit()
        conn.close()

        s = _TaskStore(str(tmp_path))
        assert run(s.max_event_seq()) == 1
        assert [e["type"] for e in run(s.events_after("t", 0))] == ["done"]
        pk = [r["name"] for r in sorted(s._conn.execute("PRAGMA table_info(events)"), key=lambda r: r["pk"]) if r["pk"]]
        assert pk == ["task_id", "seq"]

    def test_get_store_switches_connection_with_config(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TASKS_CONFIG, "data_dir", str(tmp_path / "a"))
        first = store_mod.get_store()
        monkeypatch.setattr(TASKS_CONFIG, "data_dir", str(tmp_path / "b"))
        second = store_mod.get_store()
        assert first is not second
        assert second.path == str(tmp_path / "b" / "tasks.db")
        run(second.create_task("x", "generate", {}))
        assert run(second.get_task("x")) is not None


class TestTaskBus:
    def test_emit_replay_live_and_terminal_gc(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TASKS_CONFIG, "data_dir", str(tmp_path))
        b = _TaskBus()

        async def scenario() -> None:
            await get_store().create_task("t9", "generate", {}, images_total=1)
            await b.emit("t9", "status", {"status": "queued"})
            replay, queue, terminal = await b.subscribe("t9", 0)
            assert [e["type"] for e in replay] == ["status"]
            assert terminal is False
            await b.emit("t9", "progress", {"step": 1})
            # 进度类事件零写库：回放只在内存里，库里此刻一行都没有
            assert await get_store().events_after("t9", 0) == []
            live = await asyncio.wait_for(queue.get(), 1)
            assert live["type"] == "progress" and live["seq"] > replay[0]["seq"]
            assert live["data"] == {"step": 1}
            await get_store().update_task("t9", status="done", finished_at=2.0)
            await b.emit("t9", "done", {"status": "done"})
            live = await asyncio.wait_for(queue.get(), 1)
            assert live["type"] == "done"
            b.unsubscribe("t9", queue)
            # 终态且无人订阅：内存态整体回收
            assert "t9" not in b._subs and "t9" not in b._states
            # 新订阅者回放走库：终态 emit 已落库，只剩 done 一行，terminal 标记为真
            replay2, _, terminal2 = await b.subscribe("t9", 0)
            assert [e["type"] for e in replay2] == ["done"]
            assert terminal2 is True

        asyncio.run(scenario())

    def test_emit_drops_oldest_when_subscriber_stalls(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # 订阅者队列封顶：停滞消费不会无限吃内存，满了挤掉最老一条（客户端凭 seq 跳变重连补看）
        monkeypatch.setattr(TASKS_CONFIG, "data_dir", str(tmp_path))
        monkeypatch.setattr(bus_mod, "_QUEUE_MAX", 2)
        b = _TaskBus()

        async def scenario() -> list[int]:
            _, queue, _ = await b.subscribe("t-stall", 0)
            for i in range(5):
                await b.emit("t-stall", "progress", {"i": i})
            return [queue.get_nowait()["data"]["i"] for _ in range(queue.qsize())]

        assert asyncio.run(scenario()) == [3, 4]

    def test_events_after_seq_skips_replayed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(TASKS_CONFIG, "data_dir", str(tmp_path))
        b = _TaskBus()

        async def scenario() -> None:
            seqs = [await b.emit("t8", "status", {}), await b.emit("t8", "progress", {})]
            replay, _, _ = await b.subscribe("t8", seqs[0])
            # 只回放 seq 严格大于断点的事件：断点自身不重复
            assert [e["seq"] for e in replay] == seqs[1:]

        asyncio.run(scenario())


class TestSseFormat:
    def test_event_three_part_format(self) -> None:
        out = _format_event({"seq": 5, "type": "progress", "data": {"step": 2}})
        assert out == 'id: 5\nevent: progress\ndata: {"step": 2}\n\n'

    def test_non_ascii_payload_stays_readable(self) -> None:
        out = _format_event({"seq": 1, "type": "error", "data": {"message": "执行失败"}})
        assert "执行失败" in out and "\n" not in out.split("data: ", 1)[1].rstrip("\n")
