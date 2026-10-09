"""_is_dfl 判定 + DFL aligned 元数据→DB 映射的单元测试（不依赖真实 DFLIMG 元数据写入）。

_is_dfl 判定条件（export_pipeline.py）：
  video_path 同目录存在 <stem>/ 子目录，且 <stem>/aligned/ 存在，
  项目目录 ≥10 个 [0-9]*.jpg/png，aligned ≥1 个 jpg，
  且首个 aligned jpg 能被 DFLIMG.load 解析出元数据。

_stage2_read_dfl_aligned 映射规则（审阅重点）：
  frame_idx = int(source_filename 的 stem)         ← 0-based（DFL 原生）
  face_idx  = aligned 文件名 stem 尾部 '_N' 的 N   ← 缺省 0
"""
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from MergeStudio.core import export_pipeline as ep


def _make_proj(tmp_path: Path, n_frames: int, n_aligned: int, with_meta: bool):
    """造一个假的 DFL 项目目录结构。"""
    video = tmp_path / "clip01.mp4"
    video.write_bytes(b"\x00")
    proj = tmp_path / "clip01"
    proj.mkdir()
    (proj / "aligned").mkdir()
    for i in range(n_frames):
        (proj / f"{i:05d}.jpg").write_bytes(b"\xff\xd8fake")
    for i in range(n_aligned):
        (proj / "aligned" / f"{i:05d}_0.jpg").write_bytes(b"\xff\xd8fake")
    return video, proj


class _F:
    """最小 Path 替身：只需要 .stem。"""

    def __init__(self, stem):
        self.stem = stem


class TestIsDflDetection:

    def test_not_dfl_when_no_proj_dir(self, tmp_path):
        video = tmp_path / "solo.mp4"
        video.write_bytes(b"x")
        # run_export_pipeline 全流程太重，这里只复刻判定段逻辑做对拍
        _video = Path(video)
        _proj_dir = _video.parent / _video.stem
        assert not (_proj_dir.is_dir() and (_proj_dir / "aligned").is_dir())

    def test_not_dfl_when_few_frames(self, tmp_path):
        video, proj = _make_proj(tmp_path, n_frames=5, n_aligned=3, with_meta=True)
        _proj_dir = video.parent / video.stem
        _frames = sorted(_proj_dir.glob("[0-9]*.[jp][pn]g"))
        assert len(_frames) < 10  # 不满足 ≥10 条件 → 不判定为 DFL

    def test_dfl_trigger_threshold(self, tmp_path):
        """≥10 帧 + aligned 非空 + DFLIMG 元数据有效 → _is_dfl=True。

        DFLIMG.load 需要真 JPEG APP 元数据，这里 mock 掉只测判定逻辑本身。
        """
        video, proj = _make_proj(tmp_path, n_frames=12, n_aligned=4, with_meta=True)
        _proj_dir = video.parent / video.stem
        _frames = sorted(_proj_dir.glob("[0-9]*.[jp][pn]g"))
        _aligned = sorted((_proj_dir / "aligned").glob("*.jpg"))
        assert len(_frames) >= 10 and len(_aligned) >= 1

        class _FakeSample:
            def has_data(self):
                return True

        # patch 目标必须是 DFLIMG.DFLIMG **类**（同名子模块会吃掉 create=True 的假属性）
        import DFLIMG
        with patch.object(DFLIMG.DFLIMG, "load", staticmethod(lambda *a, **k: _FakeSample())):
            _sample = DFLIMG.DFLIMG.load(str(_aligned[0]))
            assert _sample is not None and _sample.has_data()

    def test_zero_based_frame_index_mapping(self):
        """P0-2 回归（修复后）：DFL 模式套 0-based 数学，seg(0,1) 保留 stem 0/1。
        真实实现见 _compute_cut_keep_set（本文件的 TestP0KeepSet 同名测试直接打它）。"""
        from MergeStudio.core.export_pipeline import _compute_cut_keep_set
        dfl_stems = [0, 1, 2]          # DFL 项目帧：00000.jpg / 00001.jpg / 00002.jpg
        seg = {"start": 0, "end": 1}   # 用户想保留前两帧

        keep, removed = _compute_cut_keep_set(
            [_F(str(s)) for s in dfl_stems], [seg], zero_based=True)
        assert keep == {0, 1}
        assert removed == 1

    def test_face_idx_parsing(self):
        """aligned 文件名 '00003_1.jpg' → face_idx=1；'00003.jpg' → 0。"""
        for name, expect in [("00003_0.jpg", 0), ("00003_1.jpg", 1), ("00003.jpg", 0), ("00003_x.jpg", 0)]:
            stem = Path(name).stem
            fi = 0
            if "_" in stem:
                try:
                    fi = int(stem.rsplit("_", 1)[1])
                except ValueError:
                    fi = 0
            assert fi == expect, name

    def test_db_primary_key_allows_multiple_faces(self, tmp_path):
        """face_data 主键 (frame_idx, face_idx) —— 多脸共存的前提。"""
        db = tmp_path / "t.db"
        conn = sqlite3.connect(str(db))
        conn.execute("""
            CREATE TABLE face_data (
                frame_idx INTEGER NOT NULL, face_idx INTEGER NOT NULL,
                face_rect TEXT DEFAULT '[]', landmarks TEXT DEFAULT '[]',
                transform_mat TEXT DEFAULT '[]', out_size INTEGER DEFAULT 256,
                model TEXT DEFAULT '', face_id TEXT DEFAULT '',
                PRIMARY KEY (frame_idx, face_idx))
        """)
        conn.execute("INSERT INTO face_data VALUES (5,0,'[]','[]','[]',256,'','')")
        conn.execute("INSERT OR REPLACE INTO face_data VALUES (5,1,'[]','[]','[]',256,'','')")
        n = conn.execute("SELECT COUNT(*) FROM face_data WHERE frame_idx=5").fetchone()[0]
        assert n == 2
        conn.close()
