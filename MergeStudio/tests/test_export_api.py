"""routes_export 状态机测试：start / progress / stale-job 自动复位 / cancel / reset。

用 FastAPI TestClient + monkeypatch 把真实管线替换成可控的假函数，
不加载模型、不碰 GPU。注意两点：
  - routes_export 在 _run() 内部延迟 import core.export_pipeline，
    所以必须 patch core 模块上的名字；
  - TestClient 实例有内部属性（如 _state），自定义状态挂在独立 holder 上。
"""
import threading
import time

import pytest
from fastapi.testclient import TestClient


class _Holder:
    """测试状态载体，避免与 TestClient 内部属性撞名。"""


@pytest.fixture()
def env(monkeypatch, tmp_path):
    from MergeStudio.api.server import create_app
    from MergeStudio.core import export_pipeline as ep_mod
    from MergeStudio.api import routes_export as re_mod

    fake_video = tmp_path / "fake.mp4"
    fake_video.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # 只需 exists()

    h = _Holder()
    h.calls = []
    h.gate = threading.Event()
    h.started = threading.Event()

    def fake_pipeline(**kwargs):
        h.calls.append(kwargs)
        h.started.set()
        h.gate.wait(timeout=30)  # 模拟一段实际工作
        cb = kwargs.get("progress_callback")
        if cb:
            cb(6, "Encoding video", 1.0, "done")

    monkeypatch.setattr(ep_mod, "run_export_pipeline", fake_pipeline)

    app = create_app()
    with TestClient(app) as c:
        def _start(payload=None):
            body = {"video_path": str(fake_video), "num_workers": 0, **(payload or {})}
            return c.post("/api/export/start", json=body)

        h.client = c
        h.start = _start
        yield h

    # ---- 收尾：放行残留线程，恢复状态机，保证用例间隔离 ----
    h.gate.set()
    for _ in range(100):
        t = re_mod._export_thread
        if (t is None or not t.is_alive()) and not re_mod._export_job["running"]:
            break
        time.sleep(0.05)
    re_mod._export_stop.clear()
    re_mod._export_job.update({"running": False, "stage": 0, "stage_name": "",
                               "progress": 0.0, "message": "", "tick": 0})
    re_mod._export_thread = None


def _wait_started(h, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if h.started.is_set():
            return True
        time.sleep(0.05)
    return False


def test_start_ok(env):
    r = env.start()
    assert r.status_code == 200, r.text
    assert r.json()["status"] == "started"
    assert _wait_started(env)
    env.gate.set()


def test_progress_endpoint_shape(env):
    r = env.client.get("/api/export/progress/default")
    assert r.status_code == 200
    for key in ("running", "stage", "stage_name", "progress", "message", "tick"):
        assert key in r.json()


def test_double_start_rejected(env):
    r1 = env.start()
    assert r1.status_code == 200
    assert _wait_started(env)
    r2 = env.start()
    assert r2.status_code == 400  # already running
    env.gate.set()


def test_worker_cap_applied(env):
    """P0-6 回归锚：前端显式传 99 → 服务端 min(99, 4) = 4。"""
    r = env.start({"num_workers": 99})
    assert r.status_code == 200
    assert _wait_started(env)
    assert env.calls and env.calls[-1]["num_workers"] == 4
    env.gate.set()


def test_start_missing_video_400(env):
    r = env.client.post("/api/export/start", json={"video_path": "Z:/no/such.mp4"})
    assert r.status_code == 400


def test_cancel_when_idle(env):
    r = env.client.post("/api/export/cancel/default")
    assert r.status_code == 200
    assert r.json()["status"] == "not_running"


def test_reset_when_idle(env):
    r = env.client.post("/api/export/reset")
    assert r.status_code == 200
    assert r.json()["status"] == "reset"


def test_cancel_running_flow(env):
    env.start()
    assert _wait_started(env)
    r = env.client.post("/api/export/cancel/default")
    assert r.status_code == 200
    assert r.json()["status"] in ("cancelling", "cancelled")
    env.gate.set()  # 假管线醒来后 progress cb 抛 StopRequested → Cancelled
    for _ in range(100):
        if not env.client.get("/api/export/progress/default").json()["running"]:
            break
        time.sleep(0.05)
    assert env.client.get("/api/export/progress/default").json()["running"] is False


def test_stale_job_auto_reset(env):
    """P0-4 回归锚：running=True 但线程已死 → start 自动复位而不是 400。"""
    from MergeStudio.api import routes_export as re_mod
    re_mod._export_job.update({"running": True, "message": "ghost"})
    re_mod._export_thread = None
    r = env.start()
    assert r.status_code == 200, f"stale job should auto-reset, got {r.status_code}: {r.text}"
    env.gate.set()
