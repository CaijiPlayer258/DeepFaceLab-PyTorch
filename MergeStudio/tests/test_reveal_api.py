"""/api/export/reveal 端点测试：路径安全边界（不真开 explorer，mock Popen）。"""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from MergeStudio.api import routes_reveal as rr
    from MergeStudio.api.server import create_app

    # workspace root 指到 tmp
    monkeypatch.setattr(rr, "_workspace_root", lambda: tmp_path)

    opened = []
    monkeypatch.setattr("subprocess.Popen", lambda args, **k: opened.append(args))

    app = create_app()
    with TestClient(app) as c:
        c._opened = opened
        c._tmp = tmp_path
        yield c


def test_reveal_inside_workspace_ok(client):
    f = client._tmp / "result.mp4"
    f.write_bytes(b"x")
    r = client.post("/api/export/reveal", json={"path": str(f)})
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert client._opened and "/select," in client._opened[0]


def test_reveal_outside_workspace_rejected(client):
    r = client.post("/api/export/reveal", json={"path": r"C:\Windows\explorer.exe"})
    assert r.status_code == 200
    assert r.json()["status"] == "error"
    assert not client._opened


def test_reveal_missing_file_rejected(client):
    r = client.post("/api/export/reveal", json={"path": str(client._tmp / "nope.mp4")})
    assert r.json()["status"] == "error"
    assert not client._opened
