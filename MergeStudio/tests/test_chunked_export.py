"""分片导出（chunked）单元测试：规划数学 / 切帧重命名 / 编码重编号 / concat 列表。"""
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from MergeStudio.core.chunked_export import (
    plan_chunks, extract_frame_range, encode_chunk, concat_chunks, _disk_free_bytes,
)


class TestPlanChunks:
    """分片规划：按缓冲盘空余空间决定单片帧数。"""

    def test_small_video_single_chunk(self):
        # 1000 帧、空间充裕 → 单片
        with patch("MergeStudio.core.chunked_export._disk_free_bytes", return_value=500 * 1024**3):
            chunks = plan_chunks(1000, Path("X:/buf"), free_frac=0.25, max_chunk_gb=20)
        assert chunks == [(0, 999)]

    def test_split_by_free_fraction(self):
        # 剩余 40GB × 25% = 10GB - 5GB 余量 = 5GB 预算 ≈ 1462 帧（3.5MB/帧）
        with patch("MergeStudio.core.chunked_export._disk_free_bytes", return_value=40 * 1024**3):
            chunks = plan_chunks(3000, Path("X:/buf"), free_frac=0.25, max_chunk_gb=20)
        assert len(chunks) >= 2
        # 覆盖完整且不重叠
        flat = [f for c in chunks for f in range(c[0], c[1] + 1)]
        assert flat == list(range(3000))

    def test_max_chunk_cap(self):
        # 空间巨大（8TB）→ 预算 = min(8TB×50%, 20GB) - 5GB 余量 = 15GB ≈ 4380 帧
        with patch("MergeStudio.core.chunked_export._disk_free_bytes", return_value=8 * 1024**4):
            chunks = plan_chunks(20000, Path("X:/buf"), free_frac=0.5, max_chunk_gb=20)
        n0 = chunks[0][1] - chunks[0][0] + 1
        assert 4000 < n0 <= 4400  # 15GB / 3.5MB ≈ 4380
        assert len(chunks) >= 4

    def test_min_frames_floor(self):
        # 空间极小（预算<0）→ 每片仍至少 min_frames=300，不死循环
        with patch("MergeStudio.core.chunked_export._disk_free_bytes", return_value=1 * 1024**3):
            chunks = plan_chunks(700, Path("X:/buf"), free_frac=0.1, max_chunk_gb=20)
        n0 = chunks[0][1] - chunks[0][0] + 1
        assert n0 == 300

    def test_real_disk_detection(self):
        # 真实磁盘可用（temp 目录所在盘）
        free = _disk_free_bytes(Path("."))
        assert free > 0


class TestExtractRename:
    """切帧后的 tmp → 全局帧号重命名逻辑（离线模拟 ffmpeg 产物）。"""

    def _fake_ffmpeg(self, tmp_path):
        """伪造 ffmpeg 输出：tmp_00000001.jpg ... tmp_0000000N.jpg"""
        def fake_run(cmd, capture_output=True):
            # 从 cmd 里解析 -vf select 的区间
            vf = cmd[cmd.index("-vf") + 1]
            import re
            m = re.search(r"between\(n,(\d+),(\d+)\)", vf)
            s0, e0 = int(m.group(1)), int(m.group(2))
            for i in range(s0 + 1, e0 + 2):
                (tmp_path / f"tmp_{i:08d}.jpg").write_bytes(b"x")
            class R:
                returncode = 0
                stderr = b""
            return R()
        return fake_run

    def test_rename_to_global_indices(self, tmp_path, monkeypatch):
        out = tmp_path / "frames"
        monkeypatch.setattr("subprocess.run", self._fake_ffmpeg(out))
        n = extract_frame_range("video.mp4", out, 100, 104)  # 0-based [100,104] → 文件 101..105
        assert n == 5
        names = sorted(f.name for f in out.glob("*.jpg"))
        assert names == [f"{i:08d}.jpg" for i in range(101, 106)]
        assert not list(out.glob("tmp_*"))


class TestEncodeChunkRenumber:
    def test_gap_free_renumber(self, tmp_path):
        # 模拟合并产物：全局帧号有空洞（00101, 00103）
        for stem in ("00101", "00103"):
            (tmp_path / f"{stem}.jpg").write_bytes(b"x")
        (tmp_path / "concat_marker").write_text("")  # 干扰项（非图片）
        # encode_chunk 会调 ffmpeg —— mock 掉
        import subprocess as sp
        from unittest.mock import MagicMock
        with patch.object(sp, "run") as mr:
            mr.return_value = MagicMock(returncode=0, stderr=b"")
            encode_chunk(tmp_path, tmp_path / "out.mp4", "src.mp4", "h264_nvenc", 29.97)
        # 帧已重编号为 1..2（gap-free）
        stems = sorted(f.stem for f in tmp_path.glob("*.jpg"))
        assert stems == ["00000001", "00000002"]


class TestConcatList:
    def test_single_chunk_direct(self, tmp_path):
        c = tmp_path / "only.mp4"
        c.write_bytes(b"v")
        out = tmp_path / "final.mp4"
        import subprocess as sp
        from unittest.mock import MagicMock
        with patch.object(sp, "run") as mr:
            mr.return_value = MagicMock(returncode=0, stderr=b"")
            ok = concat_chunks([c], out, "src.mp4")
        assert ok
        mr.assert_called_once()
        cmd = mr.call_args.args[0]
        assert "concat" not in cmd[:4]  # 单片不走 concat demuxer

    def test_multi_chunk_concat_list(self, tmp_path):
        files = []
        for i in range(3):
            f = tmp_path / f"c{i}.mp4"
            f.write_bytes(b"v")
            files.append(f)
        out = tmp_path / "final.mp4"
        import subprocess as sp
        from unittest.mock import MagicMock
        with patch.object(sp, "run") as mr:
            mr.return_value = MagicMock(returncode=0, stderr=b"")
            ok = concat_chunks(files, out, "src.mp4")
        assert ok
        cmd = mr.call_args.args[0]
        assert "concat" in cmd[:5] and "-safe" in cmd[:6]  # 多片走 concat demuxer
        lst = tmp_path / ".final_concat.txt"
        assert lst.exists()
        content = lst.read_text()
        assert content.count("file '") == 3
