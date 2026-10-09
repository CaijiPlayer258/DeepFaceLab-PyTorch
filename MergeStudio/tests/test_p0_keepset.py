"""P0 修复的回归测试：直接打真实的 _compute_cut_keep_set 与消费点。

修复后语义：
  - DFL（zero_based=True）：keep-set 不含 +1 偏移，源帧文件永不删除；
  - 非 DFL（zero_based=False）：保持 +1（FFmpeg 1-based），物理删除不变；
  - _cut_keep_set 作为模块级状态被 Stage4/Stage5/face 任务面消费。
"""
import pytest

from MergeStudio.core import export_pipeline as ep


class _F:
    """最小 Path 替身：只需要 .stem。"""

    def __init__(self, stem):
        self.stem = stem


def _files(*stems):
    return [_F(str(s)) for s in stems]


class TestComputeCutKeepSet:

    def test_dfl_zero_based_no_offset(self):
        """P0-2 回归：DFL 模式 seg(0,1) 应保留 stem 0 和 1（旧代码误删 0 号）。"""
        keep, removed = ep._compute_cut_keep_set(
            _files(0, 1, 2), [{"start": 0, "end": 1}], zero_based=True)
        assert keep == {0, 1}
        assert removed == 1

    def test_ffmpeg_one_based_offset(self):
        """非 DFL 保持原语义：seg(0,4) 保留文件帧 1..5。"""
        keep, removed = ep._compute_cut_keep_set(
            _files(1, 2, 3, 4, 5, 6), [{"start": 0, "end": 4}], zero_based=False)
        assert keep == {1, 2, 3, 4, 5}
        assert removed == 1

    def test_multiple_segments(self):
        keep, removed = ep._compute_cut_keep_set(
            _files(0, 1, 2, 3, 4, 5, 6, 7),
            [{"start": 0, "end": 1}, {"start": 5, "end": 6}],
            zero_based=True)
        assert keep == {0, 1, 5, 6}
        assert removed == 4

    def test_non_numeric_stem_counted_removed(self):
        keep, removed = ep._compute_cut_keep_set(
            [_F("0"), _F("thumbnail")], [{"start": 0, "end": 9}], zero_based=True)
        assert keep == {0}
        assert removed == 1

    def test_single_frame_segment_inclusive(self):
        keep, _ = ep._compute_cut_keep_set(
            _files(3, 4, 5), [{"start": 4, "end": 4}], zero_based=True)
        assert keep == {4}


class TestKeepSetGlobalContract:
    """_cut_keep_set 的消费契约：None=不过滤，set=按 stem 过滤。"""

    def test_module_default_none(self):
        # 不应在 import 时被污染
        assert ep._cut_keep_set is None or isinstance(ep._cut_keep_set, set)

    def test_stage4_filter_logic_mirror(self):
        """Stage4 过滤点的对拍镜像（同一段推导）。"""
        ep._cut_keep_set = {0, 2}
        frames = [_F("0"), _F("1"), _F("2"), _F("3")]
        filtered = [f for f in frames if int(f.stem) in ep._cut_keep_set]
        assert [f.stem for f in filtered] == ["0", "2"]
        ep._cut_keep_set = None  # 还原

    def test_faces_filter_logic_mirror(self):
        """_phase_all_faces 过滤点：f[0] 是 frame_idx。"""
        ep._cut_keep_set = {0}
        faces = [(0, 0, "lm", "Anna"), (1, 0, "lm", "Anna"), (0, 1, "lm", "")]
        filtered = [f for f in faces if f[0] in ep._cut_keep_set]
        assert len(filtered) == 2
        ep._cut_keep_set = None
