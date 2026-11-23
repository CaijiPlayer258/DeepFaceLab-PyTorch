# -*- coding: utf-8 -*-
"""打一个"纯代码"发布包，供自动更新使用。

只收代码与文档文本；图片、模型、运行时、工作区一律不收，所以包体很小。
    python app/tools/make_release_zip.py            # 只打包
    python app/tools/make_release_zip.py --upload   # 打包并挂到最新 release
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ("app", "desktop", "tools", "version.txt", "README.md", "README_EN.md", "LICENSE")
SKIP_DIRS = {"__pycache__", "node_modules", ".git", "dist", "workspace", "runtime", ".backup"}
SKIP_SUFFIX = {".pth", ".dfm", ".onnx", ".engine", ".npz", ".mp4", ".zip", ".7z", ".exe", ".dll", ".pyd"}
# 整个 app/docs 都不进包：它是开发文档 + 旧 Qt 代码参考（legacy-reference），
# 约 16 MB，其中一半是会话对比图和旧界面截图。运行时没有任何代码 import 它，
# 只有注释里引用路径 —— 最终用户不需要，更新时白下载十几兆。
# 需要查文档的人看仓库本身。
#   app/tools/accept | bench | shots：验收脚本、基准、截图工具，外加 _probe 里的调试产物
#   （dbg_frames.png 一张就 0.8 MB）。这些是开发用的，server 不 import 它们。
SKIP_PATHS = ("app/docs", "app/tools/accept", "app/tools/bench", "app/tools/shots",
              "app/tools/audit")

# 仍保留这几条的图片过滤：万一以后有图片放回 app/web 也别把渲染图打进包
SKIP_IMAGE_DIRS = ("app/web/img", "desktop/build")


def want(rel: Path) -> bool:
    if rel.suffix.lower() in SKIP_SUFFIX:
        return False
    if any(part in SKIP_DIRS for part in rel.parts):
        return False
    posix = rel.as_posix()
    if any(posix == d or posix.startswith(d + "/") for d in SKIP_PATHS):
        return False
    if any(posix.startswith(d) for d in SKIP_IMAGE_DIRS) and rel.suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".webp"}:
        return False
    return True


def build(out: Path) -> tuple[Path, int, int]:
    files = 0
    total = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for top in INCLUDE:
            p = ROOT / top
            if p.is_file():
                zf.write(p, top)
                files += 1
                total += p.stat().st_size
            elif p.is_dir():
                for f in p.rglob("*"):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(ROOT)
                    if not want(rel):
                        continue
                    zf.write(f, rel.as_posix())
                    files += 1
                    total += f.stat().st_size
    return out, files, total


def upload(zip_path: Path) -> str:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        return "缺少 GITHUB_TOKEN，跳过上传"
    repo = "CaijiPlayer258/DeepFaceLab-PyTorch"
    api = f"https://api.github.com/repos/{repo}/releases/latest"
    import urllib.request

    def req(url, data=None, method="GET", ctype="application/json"):
        r = urllib.request.Request(url, data=data, method=method)
        r.add_header("Authorization", "Bearer " + token)
        r.add_header("Accept", "application/vnd.github+json")
        if data is not None:
            r.add_header("Content-Type", ctype)
        with urllib.request.urlopen(r, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))

    rel = req(api)
    upload_url = rel["upload_url"].split("{")[0]
    name = zip_path.name
    data = zip_path.read_bytes()
    out = req(f"{upload_url}?name={name}", data=data, method="POST", ctype="application/zip")
    return out.get("browser_download_url", name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "dist" / "deepfacelab-4.1-code.zip"))
    ap.add_argument("--upload", action="store_true")
    args = ap.parse_args()
    out, files, total = build(Path(args.out))
    size = out.stat().st_size
    print(f"打包完成 {out.name}: {files} 个文件, 原始 {total/1048576:.1f} MB, 压缩后 {size/1048576:.1f} MB")
    if args.upload:
        print("上传：", upload(out))


if __name__ == "__main__":
    main()
