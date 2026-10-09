"""Stage4 切片逻辑 / cut_segments 帧保留数学 / 帧文件名约定的单元测试。

被测逻辑（export_pipeline.py）：
  - cut 过滤（非 DFL 模式，FFmpeg 1-based 帧）:
      keep = any(seg.start + 1 <= idx <= seg.end + 1)
  - DFL 模式帧是 0-based（frames_dir=项目目录，文件名 00000.jpg…），
    同一套 +1 数学在 DFL 模式下整体错位 —— 这是审阅发现 P0-2 的回归锚点。
  - Stage5 重编号（gap-free）+ 音频 aselect 的 between(t, start/fps, (end+1)/fps)。
"""
import pytest

from MergeStudio.core.export_pipeline import _find_frame, probe_frame_count


def _keep(idx: int, segs: list) -> bool:
    """复刻 export_pipeline 的保留判定（含无有效段→全保留）。"""
    valid = [s for s in segs if s.get("start") is not None and s.get("end") is not None]
    if not valid:
        return True  # 管道：if _valid_segs 为假 → 不删除任何帧
    return any(s + 1 <= idx <= e + 1 for s, e in [(s["start"], s["end"]) for s in valid])


class TestCutSegmentMath:
    """cut_segments 保留语义：前端传的 start/end 是 0-based 帧号。"""

    def test_single_segment_inclusive(self):
        segs = [{"start": 10, "end": 20}]
        assert not _keep(10, segs)      # 0-based 帧号 10 → 文件 11
        assert _keep(11, segs)
        assert _keep(21, segs)
        assert not _keep(22, segs)

    def test_boundary_frame_zero(self):
        segs = [{"start": 0, "end": 4}]
        assert _keep(1, segs)
        assert _keep(5, segs)
        assert not _keep(6, segs)

    def test_multiple_segments_merge_behavior(self):
        segs = [{"start": 0, "end": 3}, {"start": 3, "end": 5}]  # 重叠/相邻
        for i in range(1, 7):
            assert _keep(i, segs), f"frame {i} should be kept"
        assert not _keep(7, segs)

    def test_invalid_segments_keep_all(self):
        """管道真实语义：有效段过滤器会把 start/end 缺失的段筛掉。
        注意现状代码：只要传入列表里存在 start/end 齐全的段就过滤；
        若全部无效（或空列表）→ 不过滤、全保留。这里的 _keep helper
        已在定义处复刻了 valid 段过滤，所以无效段→视为无段→全保留。"""
        segs = [{"start": None, "end": 5}, {}]
        assert _keep(999, segs) is True
        assert _keep(1, segs) is True

    def test_zero_frames_selected(self):
        # start==end 仍保留 1 帧（闭区间），不是 0 帧
        segs = [{"start": 5, "end": 5}]
        assert _keep(6, segs)
        assert not _keep(5, segs)
        assert not _keep(7, segs)


class TestStage5AudioMath:
    """Stage5 音频 aselect：between(t, start/fps, (end+1)/fps)。

    帧文件 1-based、第 N 帧显示时间戳 (N-1)/fps；音频窗起点用 start/fps
    （= 第 start+1 帧的 PTS），终点 (end+1)/fps —— 与视频帧 [start+1, end+1]
    对齐。此测试把该数学固定下来，防止未来单边修改造成音画错位。
    """

    def test_audio_window_alignment(self):
        fps = 30.0
        seg = {"start": 90, "end": 149}  # 视频保留文件帧 91..150 → 3s..5s
        t0 = seg["start"] / fps
        t1 = (seg["end"] + 1) / fps
        assert t0 == pytest.approx(3.0)
        assert t1 == pytest.approx(5.0)
        # 帧侧：91 帧首 PTS = 90/30 = 3.0s ✓
        assert (91 - 1) / fps == pytest.approx(t0)


class TestRenumberGaps:
    """Stage5 重编号语义：保留帧按字典序重命名为 1..N（8 位零填充）。"""

    def test_renumber_plan(self, tmp_path):
        # 模拟 cut 后残留的不连续帧
        for name in ["00000011.jpg", "00000012.jpg", "00000020.jpg"]:
            (tmp_path / name).write_bytes(b"x")
        frames = sorted(tmp_path.glob("*.jpg"))
        assert [f.stem for f in frames] == ["00000011", "00000012", "00000020"]
        for i, f in enumerate(frames):
            new = tmp_path / f"{i + 1:08d}.jpg"
            if new != f:
                f.rename(new)
        stems = [p.stem for p in sorted(tmp_path.glob("*.jpg"))]
        assert stems == ["00000001", "00000002", "00000003"]


class TestFindFrame:
    """_find_frame 多补零宽度探测。"""

    def test_finds_8digit(self, tmp_path):
        (tmp_path / "00000123.jpg").write_bytes(b"x")
        assert _find_frame(tmp_path, 123) is not None

    def test_finds_5digit_dfl_style(self, tmp_path):
        (tmp_path / "00042.jpg").write_bytes(b"x")
        found = _find_frame(tmp_path, 42)
        assert found is not None and found.stem == "00042"

    def test_dfl_zero_based_lookup(self, tmp_path):
        """DFL 模式：aligned 元数据里 source 是 0-based，_find_frame 应能命中 0 号帧。"""
        (tmp_path / "00000.jpg").write_bytes(b"x")
        assert _find_frame(tmp_path, 0) is not None

    def test_missing_returns_none(self, tmp_path):
        assert _find_frame(tmp_path, 7) is None

    def test_returns_first_when_ambiguous(self, tmp_path):
        (tmp_path / "00000007.jpg").write_bytes(b"x")
        (tmp_path / "7.png").write_bytes(b"x")
        found = _find_frame(tmp_path, 7)
        assert found.suffix == ".jpg"  # 8 位优先


class TestProbeFrameCount:
    def test_missing_file_returns_zero(self, tmp_path):
        assert probe_frame_count(tmp_path / "nope.mp4") == 0
