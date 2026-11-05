"""SQLite 任务持久化层。"""

import asyncio
import json
import os
import sqlite3
import threading
import time
from typing import Any, TypedDict, cast

from comfyui_anima_api.config import TASKS_CONFIG

# 事件表只存终态事件行（进度类走总线内存，见 bus.py）；seq 由总线全局分配，
# 复合主键按 (task_id, seq) 建索引天然覆盖回放查询
_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS events (
    task_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    type TEXT NOT NULL,
    data TEXT NOT NULL,
    ts REAL NOT NULL,
    PRIMARY KEY (task_id, seq)
)
"""

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,
    status TEXT NOT NULL,
    params TEXT NOT NULL,
    prompt_id TEXT,
    images_total INTEGER NOT NULL DEFAULT 0,
    images_done INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    updated_at REAL NOT NULL
);
{_EVENTS_DDL};
CREATE TABLE IF NOT EXISTS task_images (
    task_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    filename TEXT NOT NULL,
    subfolder TEXT NOT NULL DEFAULT '',
    type TEXT NOT NULL DEFAULT 'output',
    prompt_id TEXT,
    UNIQUE(task_id, idx)
);
"""

TERMINAL_STATUSES = ("done", "error", "cancelled")


class _TaskRow(TypedDict):
    id: str
    type: str
    status: str
    params: dict[str, Any]
    prompt_id: str | None
    images_total: int
    images_done: int
    error: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    updated_at: float


class EventRow(TypedDict):
    seq: int
    type: str
    data: dict[str, Any]
    ts: float


class _ImageRow(TypedDict):
    idx: int
    filename: str
    subfolder: str
    type: str
    prompt_id: str | None


class _TaskStore:
    """单连接 SQLite 封装。"""

    def __init__(self, data_dir: str) -> None:
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "tasks.db")
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._migrate_events_pk()
        self._conn.commit()

    def _migrate_events_pk(self) -> None:
        """events 采用 (task_id, seq) 复合主键；打开库时检测到单列 seq 主键的旧布局
        即重建表并原值搬运行（seq 保持单调，Last-Event-ID 语义不受影响）。"""
        cols = self._conn.execute("PRAGMA table_info(events)").fetchall()
        pk = [c["name"] for c in sorted((c for c in cols if c["pk"]), key=lambda c: c["pk"])]
        if pk == ["task_id", "seq"]:
            return
        self._conn.execute("ALTER TABLE events RENAME TO events_old")
        self._conn.execute(_EVENTS_DDL)
        self._conn.execute(
            "INSERT INTO events (task_id, seq, type, data, ts) SELECT task_id, seq, type, data, ts FROM events_old"
        )
        self._conn.execute("DROP TABLE events_old")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- tasks ----

    async def create_task(self, task_id: str, task_type: str, params: dict[str, Any], images_total: int = 0) -> None:
        now = time.time()
        def op() -> None:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO tasks (id, type, status, params, images_total, created_at, updated_at) "
                    "VALUES (?, ?, 'queued', ?, ?, ?, ?)",
                    (task_id, task_type, json.dumps(params, ensure_ascii=False), images_total, now, now),
                )
                self._conn.commit()

        await asyncio.to_thread(op)

    async def get_task(self, task_id: str) -> _TaskRow | None:
        def op() -> sqlite3.Row | None:
            with self._lock:
                return self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()

        row = await asyncio.to_thread(op)
        if row is None:
            return None
        task = dict(row)
        task["params"] = json.loads(task["params"])
        return cast(_TaskRow, task)

    async def update_task(self, task_id: str, **fields: Any) -> None:
        allowed = {"status", "prompt_id", "images_done", "images_total", "error", "started_at", "finished_at"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"unknown task columns: {sorted(bad)}")
        assignments = ", ".join(f"{k} = ?" for k in fields)

        def op() -> None:
            with self._lock:
                self._conn.execute(
                    f"UPDATE tasks SET {assignments}, updated_at = ? WHERE id = ?",
                    (*fields.values(), time.time(), task_id),
                )
                self._conn.commit()

        await asyncio.to_thread(op)

    async def unfinished_tasks(self) -> list[_TaskRow]:
        def op() -> list[sqlite3.Row]:
            with self._lock:
                placeholders = ", ".join("?" for _ in TERMINAL_STATUSES)
                return self._conn.execute(
                    f"SELECT * FROM tasks WHERE status NOT IN ({placeholders}) ORDER BY created_at",
                    TERMINAL_STATUSES,
                ).fetchall()

        rows = await asyncio.to_thread(op)
        tasks: list[_TaskRow] = []
        for row in rows:
            task = dict(row)
            task["params"] = json.loads(task["params"])
            tasks.append(cast(_TaskRow, task))
        return tasks

    # ---- events（只存终态行；进度类走总线内存，见 bus.py）----

    async def insert_event(self, task_id: str, seq: int, event_type: str, data: dict[str, Any]) -> None:
        """写入一条事件；seq 由总线全局计数器分配（非自增）。"""

        def op() -> None:
            with self._lock:
                self._conn.execute(
                    "INSERT INTO events (task_id, seq, type, data, ts) VALUES (?, ?, ?, ?, ?)",
                    (task_id, seq, event_type, json.dumps(data, ensure_ascii=False), time.time()),
                )
                self._conn.commit()

        await asyncio.to_thread(op)

    async def max_event_seq(self) -> int:
        """库中已存在的最大事件 seq；总线以此为起点计数，跨重启保持全局单调。"""

        def op() -> int:
            with self._lock:
                row = self._conn.execute("SELECT MAX(seq) FROM events").fetchone()
                return int(row[0] or 0)

        return await asyncio.to_thread(op)

    async def events_after(self, task_id: str, after_seq: int) -> list[EventRow]:
        def op() -> list[sqlite3.Row]:
            with self._lock:
                return self._conn.execute(
                    "SELECT seq, type, data, ts FROM events WHERE task_id = ? AND seq > ? ORDER BY seq",
                    (task_id, after_seq),
                ).fetchall()

        rows = await asyncio.to_thread(op)
        return [
            cast(EventRow, {"seq": r["seq"], "type": r["type"], "data": json.loads(r["data"]), "ts": r["ts"]})
            for r in rows
        ]

    # ---- task_images ----

    async def add_image(self, task_id: str, image: dict[str, Any], prompt_id: str | None) -> int:
        def op() -> int:
            with self._lock:
                idx = (
                    self._conn.execute("SELECT COUNT(*) FROM task_images WHERE task_id = ?", (task_id,)).fetchone()[0]
                    + 1
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO task_images (task_id, idx, filename, subfolder, type, prompt_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        task_id,
                        idx,
                        image["filename"],
                        image.get("subfolder", ""),
                        image.get("type", "output"),
                        prompt_id,
                    ),
                )
                done = self._conn.execute("SELECT COUNT(*) FROM task_images WHERE task_id = ?", (task_id,)).fetchone()[
                    0
                ]
                self._conn.execute(
                    "UPDATE tasks SET images_done = ?, updated_at = ? WHERE id = ?", (done, time.time(), task_id)
                )
                self._conn.commit()
                return idx

        return await asyncio.to_thread(op)

    async def images(self, task_id: str) -> list[_ImageRow]:
        def op() -> list[sqlite3.Row]:
            with self._lock:
                return self._conn.execute(
                    "SELECT idx, filename, subfolder, type, prompt_id FROM task_images WHERE task_id = ? ORDER BY idx",
                    (task_id,),
                ).fetchall()

        return [cast(_ImageRow, dict(r)) for r in await asyncio.to_thread(op)]

    async def image(self, task_id: str, idx: int) -> _ImageRow | None:
        def op() -> sqlite3.Row | None:
            with self._lock:
                return self._conn.execute(
                    "SELECT idx, filename, subfolder, type, prompt_id FROM task_images WHERE task_id = ? AND idx = ?",
                    (task_id, idx),
                ).fetchone()

        row = await asyncio.to_thread(op)
        return None if row is None else cast(_ImageRow, dict(row))


_STORE: _TaskStore | None = None
_STORE_PATH: str | None = None


def get_store() -> _TaskStore:
    global _STORE, _STORE_PATH
    path = os.path.join(TASKS_CONFIG.data_dir, "tasks.db")
    if _STORE is None or path != _STORE_PATH:
        if _STORE is not None:
            _STORE.close()
        _STORE = _TaskStore(TASKS_CONFIG.data_dir)
        _STORE_PATH = path
    return _STORE
