# -*- coding: utf-8 -*-
r"""人脸数据库后端验收（app/tools/accept/test_faces_api.py）。

临时目录里造项目，不碰用户 workspace 真实数据。运行：
    runtime\server\Scripts\python.exe app\tools\test_faces_api.py

覆盖：
  * 90 帧视频 + 帧图项目；提取 3 帧（1 / 2 / 1 张脸）与无人脸帧
  * 文件与 index 一致、PNG 尺寸 = DFM 输入尺寸、缩略图 128x128 + 缓存头
  * 跨组投放文件不动；回收站 trash/restore/delete
  * groups 幂等 + orphan 保留
  * 路径穿越 400
  * 10 个 move 并发后 index 可解析、数量守恒
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[3]
APP = REPO / "app"
sys.path.insert(0, str(APP))
sys.path.append(str(APP / "engine"))

import cv2
import numpy as np

import tempfile
BASE = Path(tempfile.mkdtemp(prefix="dfl_faces_api_"))
PROJ = BASE / "proj"
PROJ.mkdir(parents=True, exist_ok=True)
print("temp project: %s" % PROJ, flush=True)

PASS, FAIL = [], []


def check(cond, name, detail=""):
    if cond:
        PASS.append(name)
        print("PASS  %s" % name, flush=True)
    else:
        FAIL.append(name + (" :: " + str(detail) if detail else ""))
        print("FAIL  %s  %s" % (name, detail), flush=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ------------------------------------------------------------
# 1) 造 90 帧视频 + 90 张帧图
# ------------------------------------------------------------
COLORS = {
    10: (10, 10, 200),      # 红：1 张脸
    15: (10, 150, 10),      # 绿：1 张脸
    40: (200, 10, 10),      # 蓝：2 张脸
    70: (150, 150, 10),     # 青：1 张脸
}
for i in range(90):
    img = np.full((360, 640, 3), 30, np.uint8)
    img[:] = COLORS.get(i, (30, 30, 30))
    cv2.imwrite(str(PROJ / ("%06d.jpg" % i)), img)

VIDEO = PROJ / "sample90.mp4"
video_ok = False
try:
    import imageio_ffmpeg
    import subprocess
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    for codec in ("libx264", "mpeg4"):
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-framerate", "30",
               "-i", str(PROJ / "%06d.jpg"), "-c:v", codec,
               "-pix_fmt", "yuv420p", str(VIDEO)]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode == 0 and VIDEO.is_file() and VIDEO.stat().st_size > 0:
            video_ok = True
            break
except Exception as exc:
    print("video ffmpeg failed: %r" % (exc,), flush=True)
if not video_ok:
    writer = cv2.VideoWriter(str(VIDEO), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 360))
    for i in range(90):
        writer.write(cv2.imread(str(PROJ / ("%06d.jpg" % i))))
    writer.release()
    video_ok = VIDEO.is_file() and VIDEO.stat().st_size > 0
check(video_ok, "90 帧视频已生成")
cap = cv2.VideoCapture(str(VIDEO))
check(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) == 90, "视频帧数 = 90",
      cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# ------------------------------------------------------------
# 2) 检测桩：按帧的底色决定 0/1/2 张脸（存储/API 验收不依赖 GPU 检测精度）
# ------------------------------------------------------------
import server._faces as faces_store

BOXES = {
    10: [(80, 60, 240, 240)],
    15: [(350, 60, 510, 240)],
    40: [(40, 80, 200, 260), (420, 80, 580, 260)],
    70: [(220, 100, 380, 260)],
}


def fake_detect(img, detector=""):
    patch = img[0:8, 0:8].reshape(-1, 3).mean(axis=0)
    b, g, r = [float(x) for x in patch]
    frame = None
    if r > 150 and g < 80 and b < 80:
        frame = 10
    elif g > 100 and r < 80 and b < 80:
        frame = 15
    elif b > 150 and g < 80 and r < 80:
        frame = 40
    elif b > 100 and g > 100 and r < 80:
        frame = 70
    out = []
    for (x1, y1, x2, y2) in BOXES.get(frame, []):
        w, h = x2 - x1, y2 - y1
        kps = [
            [x1 + 0.30 * w, y1 + 0.35 * h],
            [x1 + 0.70 * w, y1 + 0.35 * h],
            [x1 + 0.50 * w, y1 + 0.55 * h],
            [x1 + 0.35 * w, y1 + 0.75 * h],
            [x1 + 0.65 * w, y1 + 0.75 * h],
        ]
        out.append({"box": [x1, y1, x2, y2], "kps": kps, "score": 0.99})
    return out


faces_store.detect_faces_in_image = fake_detect

# ------------------------------------------------------------
# 3) 起真实的 FastAPI（线程里跑 uvicorn），HTTP 验收
# ------------------------------------------------------------
import uvicorn
from server.app import app as fastapi_app

PORT = free_port()
BASE_URL = "http://127.0.0.1:%d" % PORT
config = uvicorn.Config(fastapi_app, host="127.0.0.1", port=PORT, log_level="error")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
session = requests.Session()
session.trust_env = False
for _ in range(100):
    try:
        session.get(BASE_URL + "/api/state", timeout=1)
        break
    except Exception:
        time.sleep(0.1)


def api(method, path, **kw):
    kw.setdefault("timeout", 120)
    r = session.request(method, BASE_URL + path, **kw)
    try:
        body = r.json()
    except Exception:
        body = None
    return r, body


P = str(PROJ)
DFM_A = "张元英"
DFM_B = "阿証-416"

# ------------------------------------------------------------
# 4) 空库 + groups 幂等
# ------------------------------------------------------------
r, d = api("GET", "/api/merge/faces", params={"project": P})
check(r.status_code == 200 and d.get("ok") is True, "空库 GET 200/ok")
check(d and d.get("faces") == [] and d.get("groups") == [], "空库 faces/groups 为空")
check(d and d.get("stats", {}).get("total") == 0, "空库 stats.total=0")

r, d = api("POST", "/api/merge/faces/groups", json={"project": P, "dfms": [DFM_A, DFM_B]})
check(r.status_code == 200 and [g["dfm"] for g in d.get("groups", [])] == [DFM_A, DFM_B],
      "groups 首次同步", d.get("groups"))
groups_first = d.get("groups")
gid_a = groups_first[0]["id"] if groups_first else ""
gid_b = groups_first[1]["id"] if len(groups_first or []) > 1 else ""

r, d = api("POST", "/api/merge/faces/groups", json={"project": P, "dfms": [DFM_A, DFM_B]})
check(r.status_code == 200 and d.get("groups") == groups_first, "groups 第二次调用完全不变")
check((PROJ / "_faces" / "index.json").is_file(), "index.json 已落盘")
check((PROJ / "_faces" / "index.json.bak").is_file(), "index.json.bak 已保留")

# ------------------------------------------------------------
# 5) 提取：384 / 384 / 512 / 未分组 / 无人脸
# ------------------------------------------------------------
r, d = api("POST", "/api/merge/faces/extract", json={"project": P, "frame": 10, "dfm": DFM_A})
check(r.status_code == 200 and d.get("ok") is True, "extract frame10 200/ok", d)
check(d and d.get("count") == 1, "frame10 提取 1 张", d and d.get("count"))
check(d and d.get("size") == 384, "frame10 尺寸 = 张元英 DFM 384", d and d.get("size"))
face1 = (d.get("faces") or [{}])[0] if d else {}
check(face1.get("group") == gid_a, "frame10 落入 g1", face1.get("group"))
check(face1.get("source", {}).get("video") == "sample90.mp4"
      and face1.get("source", {}).get("frame") == 10, "source=(sample90.mp4,10)", face1.get("source"))
p1 = PROJ / "_faces" / (face1.get("file") or "")
check(p1.is_file(), "frame10 PNG 文件存在", p1)
if p1.is_file():
    im = cv2.imread(str(p1))
    check(im is not None and im.shape[:2] == (384, 384), "frame10 PNG 384x384",
          None if im is None else im.shape)
check(str(face1.get("thumb", "")).startswith("data:image/jpeg;base64,"), "extract 返回缩略图 base64")

r, d = api("POST", "/api/merge/faces/extract", json={"project": P, "frame": 15, "dfm": DFM_A})
check(r.status_code == 200 and d.get("count") == 1, "frame15 提取 1 张")
face2 = (d.get("faces") or [{}])[0]
check(face2.get("group") == gid_a, "frame15 落入 g1")

r, d = api("POST", "/api/merge/faces/extract", json={"project": P, "frame": 40, "dfm": DFM_B})
check(r.status_code == 200 and d.get("count") == 2, "frame40 一帧提取 2 张", d and d.get("count"))
check(d and d.get("size") == 512, "未知 DFM 名回退 512", d and d.get("size"))
check(all(f.get("group") == gid_b for f in (d.get("faces") or [])), "frame40 两张都落入 g2")
face3, face4 = (d.get("faces") or [{}])[0], (d.get("faces") or [{}])[1]

r, d = api("POST", "/api/merge/faces/extract", json={"project": P, "frame": 70})
check(r.status_code == 200 and d.get("count") == 1 and d["faces"][0]["group"] == "",
      "frame70 未指定 dfm → 未分组")
face5 = (d.get("faces") or [{}])[0]

r, d = api("POST", "/api/merge/faces/extract", json={"project": P, "frame": 20})
check(r.status_code == 200 and d.get("ok") is False and d.get("reason") == "no_face",
      "无人脸 → 200 + ok:false + no_face", (r.status_code, d))

index = read_json(PROJ / "_faces" / "index.json")
check(len(index.get("faces", [])) == 5, "index 里有 5 张脸", len(index.get("faces", [])))
check([f["id"] for f in index["faces"]] == ["f_0001", "f_0002", "f_0003", "f_0004", "f_0005"],
      "id 连续 f_0001..f_0005")
check(all((PROJ / "_faces" / f["file"]).is_file() for f in index["faces"]), "5 个 PNG 都在")

# ------------------------------------------------------------
# 6) 缩略图：128x128 + 缓存头 + 304
# ------------------------------------------------------------
r = session.get(BASE_URL + "/api/merge/faces/thumb", params={"project": P, "id": face1["id"]}, timeout=30)
check(r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/jpeg"),
      "thumb 200 image/jpeg", r.status_code)
check("max-age" in (r.headers.get("Cache-Control") or ""), "thumb Cache-Control 带 max-age",
      r.headers.get("Cache-Control"))
etag = r.headers.get("ETag")
arr = np.frombuffer(r.content, np.uint8)
tim = cv2.imdecode(arr, cv2.IMREAD_COLOR)
check(tim is not None and tim.shape[:2] == (128, 128), "thumb 解码为 128x128",
      None if tim is None else tim.shape)
check((PROJ / "_faces" / "_thumb" / (face1["id"] + ".jpg")).is_file(), "缩略图缓存文件已生成")
r2 = session.get(BASE_URL + "/api/merge/faces/thumb", params={"project": P, "id": face1["id"]},
                 headers={"If-None-Match": etag or ""}, timeout=30)
check(r2.status_code == 304, "If-None-Match 命中 304", r2.status_code)

# ------------------------------------------------------------
# 7) 跨组投放：g1 两张 → g2，文件不动
# ------------------------------------------------------------
before_files = {f["id"]: (PROJ / "_faces" / f["file"]) for f in index["faces"]}
before_bytes = {k: v.stat().st_size for k, v in before_files.items()}
r, d = api("POST", "/api/merge/faces/move",
           json={"project": P, "ids": [face1["id"], face2["id"]], "group": gid_b})
check(r.status_code == 200 and d.get("updated") == 2, "move g1→g2 返回 2", d)
index = read_json(PROJ / "_faces" / "index.json")
by_id = {f["id"]: f for f in index["faces"]}
check(by_id[face1["id"]]["group"] == gid_b and by_id[face2["id"]]["group"] == gid_b,
      "index 里两张都改成 g2")
check(all(before_files[k].is_file() and before_files[k].stat().st_size == before_bytes[k]
          for k in before_files), "move 后文件路径/大小完全没变")

# ------------------------------------------------------------
# 8) 回收站：trash → restore → DELETE
# ------------------------------------------------------------
tid1, tid2 = face3["id"], face4["id"]
tfile = PROJ / "_faces" / by_id[tid1]["file"]
r, d = api("POST", "/api/merge/faces/trash", json={"project": P, "ids": [tid1, tid2]})
check(r.status_code == 200 and d.get("updated") == 2, "trash 两张 200", d)
index = read_json(PROJ / "_faces" / "index.json")
by_id = {f["id"]: f for f in index["faces"]}
check(by_id[tid1]["trashed"] is True and by_id[tid2]["trashed"] is True, "index trashed=true")
check((PROJ / "_faces" / "_trash" / by_id[tid1]["file"]).is_file(), "文件已进 _trash")
check(not (PROJ / "_faces" / by_id[tid1]["file"]).is_file(), "原位置文件已不在")
r = session.get(BASE_URL + "/api/merge/faces/thumb", params={"project": P, "id": tid1}, timeout=30)
check(r.status_code == 200, "回收站里的脸仍能取缩略图", r.status_code)

r, d = api("POST", "/api/merge/faces/restore", json={"project": P, "ids": [tid1, tid2]})
check(r.status_code == 200 and d.get("updated") == 2, "restore 两张 200")
index = read_json(PROJ / "_faces" / "index.json")
by_id = {f["id"]: f for f in index["faces"]}
check(by_id[tid1]["trashed"] is False, "index trashed=false")
check((PROJ / "_faces" / by_id[tid1]["file"]).is_file(), "文件回到 _faces/")
check(not (PROJ / "_faces" / "_trash" / by_id[tid1]["file"]).is_file(), "_trash 里已空")

del_id = tid1
del_file = PROJ / "_faces" / by_id[del_id]["file"]
r, d = api("DELETE", "/api/merge/faces", json={"project": P, "ids": [del_id]})
check(r.status_code == 200 and d.get("count") == 1, "DELETE 200 count=1", d)
index = read_json(PROJ / "_faces" / "index.json")
check(len(index["faces"]) == 4, "DELETE 后 index 剩 4 张", len(index["faces"]))
check(not del_file.exists(), "DELETE 后 PNG 消失")
check(not (PROJ / "_faces" / "_thumb" / (del_id + ".jpg")).exists(), "DELETE 后缩略图缓存消失")

# ------------------------------------------------------------
# 9) groups 幂等 + orphan
# ------------------------------------------------------------
r, d = api("POST", "/api/merge/faces/groups", json={"project": P, "dfms": [DFM_A]})
check(r.status_code == 200, "groups 少传一个 dfm 200")
gb = [g for g in d.get("groups", []) if g["id"] == gid_b]
check(len(gb) == 1 and gb[0].get("orphan") is True, "g2 变成 orphan=true", gb)
check(any(f.get("group") == gid_b for f in d.get("faces", [])), "g2 的脸还在")
r2, d2 = api("POST", "/api/merge/faces/groups", json={"project": P, "dfms": [DFM_A]})
check(d2.get("groups") == d.get("groups"), "orphan 状态再同步一次不变")
r3, d3 = api("POST", "/api/merge/faces/groups", json={"project": P, "dfms": [DFM_A, DFM_B]})
gb2 = [g for g in d3.get("groups", []) if g["id"] == gid_b]
check(len(gb2) == 1 and not gb2[0].get("orphan"), "重新选中后 orphan 复位，id 不变", gb2)

# ------------------------------------------------------------
# 10) 路径穿越 / 非法 id
# ------------------------------------------------------------
r, d = api("GET", "/api/merge/faces", params={"project": "../.."})
check(r.status_code == 400, "project=../.. → 400", r.status_code)
r, d = api("POST", "/api/merge/faces/move",
           json={"project": P, "ids": ["../f_0001.png"], "group": gid_a})
check(r.status_code == 400, "ids 带 ../ → 400", r.status_code)
r, d = api("POST", "/api/merge/faces/move",
           json={"project": P, "ids": ["C:\\Windows\\win.ini"], "group": gid_a})
check(r.status_code == 400, "ids 绝对路径 → 400", r.status_code)
r, d = api("POST", "/api/merge/faces/move",
           json={"project": P, "ids": [face1["id"]], "group": "../g1"})
check(r.status_code == 400, "group 带 ../ → 400", r.status_code)
r = session.get(BASE_URL + "/api/merge/faces/thumb", params={"project": P, "id": "../f_0001"}, timeout=30)
check(r.status_code == 400, "thumb id 带 ../ → 400", r.status_code)
r, d = api("POST", "/api/merge/faces/groups", json={"project": P, "dfms": ["../evil"]})
check(r.status_code == 400, "dfms 带 ../ → 400", r.status_code)

# ------------------------------------------------------------
# 11) 并发：10 个 move 同时打
# ------------------------------------------------------------
ids10 = [f["id"] for f in index["faces"]]
results = []


def worker(i):
    group = gid_a if i % 2 == 0 else gid_b
    rr, dd = api("POST", "/api/merge/faces/move",
                 json={"project": P, "ids": ids10, "group": group})
    results.append((rr.status_code, dd and dd.get("ok")))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join(60)
check(len(results) == 10 and all(code == 200 and ok for code, ok in results),
      "10 个并发 move 全部 200/ok", results)
try:
    idx2 = read_json(PROJ / "_faces" / "index.json")
    parsed = True
except Exception as exc:
    idx2, parsed = {}, False
    print("index parse error", repr(exc), flush=True)
check(parsed, "并发后 index.json 可解析")
check(len(idx2.get("faces", [])) == 4, "并发后脸数量守恒 = 4", len(idx2.get("faces", [])))
check(len({f["id"] for f in idx2.get("faces", [])}) == 4, "并发后 id 不重复")
check(all(f["group"] in (gid_a, gid_b) for f in idx2.get("faces", [])), "并发后 group 合法")
check(all((PROJ / "_faces" / f["file"]).is_file() for f in idx2.get("faces", [])),
      "并发后 4 个 PNG 都在")

# ------------------------------------------------------------
# 12) DFM 输入尺寸解析（读真实 .dfm，只读头部，不加载权重）
# ------------------------------------------------------------
dfm_a = REPO / "workspace" / "model" / (DFM_A + ".dfm")
if dfm_a.is_file():
    check(faces_store.dfm_input_size(dfm_a) == 384, "真实 DFM 张元英 输入尺寸=384",
          faces_store.read_onnx_input_shape(dfm_a))
else:
    print("SKIP 真实 DFM 不在（%s）" % dfm_a, flush=True)

dfm_b = REPO / "workspace" / "model" / "阿証証証証-416.dfm"
if dfm_b.is_file():
    check(faces_store.dfm_input_size(dfm_b) == 416, "真实 DFM 阿証-416 输入尺寸=416",
          faces_store.read_onnx_input_shape(dfm_b))
else:
    print("SKIP 真实 DFM 不在（%s）" % dfm_b, flush=True)

# ------------------------------------------------------------
# 汇总
# ------------------------------------------------------------
server.should_exit = True
thread.join(10)
report = {"pass": len(PASS), "fail": len(FAIL), "failures": FAIL,
          "project": str(PROJ), "pass_names": PASS}
(BASE / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
print("\nRESULT pass=%d fail=%d" % (len(PASS), len(FAIL)), flush=True)
if FAIL:
    print("FAILURES:", json.dumps(FAIL, ensure_ascii=False), flush=True)
sys.exit(0 if not FAIL else 1)