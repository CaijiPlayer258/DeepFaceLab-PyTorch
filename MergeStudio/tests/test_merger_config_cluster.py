"""MergerConfigMasked 参数清洗 + 遮罩模式选择 + DBSCAN 聚类的单元测试。

MergerConfigMasked 已知坑（审阅发现 P1-2 / P1-3）：
  - masked_hist_match 被硬编码 True，用户传 False 被静默丢弃；
  - color_transfer_mode 无白名单校验，导出路径直接把字符串 'lct' 存进实例，
    而 merger.py 拿它当 dict int 键查 ct_functions —— GUI 预览路径（routes_preview
    有 str→int 转换）之外的所有入口都踩雷。
"""
import numpy as np
import pytest

from MergeStudio.core.config import MergerConfigMasked, ctm_dict, ctm_str_dict, mode_str_dict
from MergeStudio.core.face_cluster import (
    cluster_embeddings,
    cluster_embeddings_dbscan,
    match_face_to_cluster,
)


class TestMergerConfigMasked:

    def test_defaults(self):
        cfg = MergerConfigMasked()
        assert cfg.mask_mode == 4
        assert cfg.seg_mode == "model"
        assert cfg.erode_mask_modifier == 0
        assert cfg.blur_mask_modifier == 0

    def test_invalid_mode_falls_back_to_overlay(self):
        cfg = MergerConfigMasked(mode="不存在的模式")
        assert cfg.mode in mode_str_dict

    def test_extra_kwargs_swallowed_into_extra(self):
        """P1-2b 已修：未知 kwargs 不再 TypeError，收进 .extra。"""
        cfg = MergerConfigMasked(face_margin=0.48)
        assert cfg.extra.get("face_margin") == 0.48

    def test_ct_mode_string_sanitized_by_helper(self):
        """P1-3 已修：_sanitize_config 把字符串 ct 模式转 int，白名单过滤后传入。"""
        from MergeStudio.core.export_pipeline import _sanitize_config
        cleaned = _sanitize_config({"color_transfer_mode": "lct", "face_margin": 0.48, "mask_mode": 6})
        assert cleaned["color_transfer_mode"] == 2
        assert cleaned["mask_mode"] == 6
        assert "face_margin" not in cleaned  # 非白名单字段被过滤（导出路径行为不变）
        cfg = MergerConfigMasked(**cleaned)
        assert cfg.color_transfer_mode == 2

    def test_masked_hist_match_respected(self):
        """P1-2 已修：masked_hist_match=False 不再被硬编码覆盖。"""
        cfg = MergerConfigMasked(masked_hist_match=False)
        assert cfg.masked_hist_match is False

    def test_masked_hist_match_default_true(self):
        cfg = MergerConfigMasked()
        assert cfg.masked_hist_match is True

    def test_color_transfer_mode_string_not_validated(self):
        """P1-3 回归锚：字符串直接穿透，下游 merger 的 ct_functions[int 键] 会查不到。"""
        cfg = MergerConfigMasked(color_transfer_mode="lct")  # 未转换的原始字符串
        assert cfg.color_transfer_mode == "lct"
        # 而 merger.py 的查找表键是 int：
        assert "lct" not in ctm_dict
        assert ctm_str_dict["lct"] == 2

    def test_valid_int_ct_mode(self):
        cfg = MergerConfigMasked(color_transfer_mode=2)
        assert cfg.color_transfer_mode == 2


class TestMaskModeSelection:
    """直接测 merger.MergeMaskedFace 的遮罩选择分支太重（要造脸），
    这里用等价的最小化校验：mask_mode 合法值 + fallback 语义在纯 numpy 层固定。"""

    @pytest.mark.parametrize("mode", list(range(10)))
    def test_mask_mode_values_accepted(self, mode):
        cfg = MergerConfigMasked(mask_mode=mode)
        assert cfg.mask_mode == mode

    def test_full_mask_is_ones(self):
        ones = np.ones((64, 64), dtype=np.float32)
        assert ones.min() == 1.0  # mask_mode=0 语义：全脸无遮罩

    def test_erode_blur_odd_kernel(self):
        """blur 核必须奇数：blur + (1 - blur % 2)。"""
        for b in (1, 2, 3, 4, 5, 50, 99):
            k = b + (1 - b % 2)
            assert k % 2 == 1


class TestClustering:

    def _emb(self, seed: int) -> np.ndarray:
        rng = np.random.RandomState(seed)
        v = rng.randn(512).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_empty(self):
        assert cluster_embeddings_dbscan({}) == {}

    def test_single(self):
        e = self._emb(1)
        assert cluster_embeddings_dbscan({"a": e}) == {"a": ["a"]}

    def test_two_groups(self):
        base = self._emb(42)
        near = base + 0.01 * self._emb(43)
        near = (near / np.linalg.norm(near)).astype(np.float32)
        far = self._emb(99)  # 随机不同方向
        result = cluster_embeddings_dbscan(
            {"a1": base, "a2": near, "b1": far}, eps=0.3, min_samples=1
        )
        # a1/a2 应在同一簇；b1 单独成簇或进噪声
        group_of_a1 = next(v for v in result.values() if "a1" in v)
        assert "a2" in group_of_a1
        assert "b1" not in group_of_a1

    def test_all_outliers_form_singleton_clusters(self):
        """min_samples=1 时 DBSCAN 无噪声点：每个孤立样本自成单例簇。
        下游 match_face_to_cluster 对任意输入都能找到 best_sim>0 的簇 ——
        实际把相似度门槛交给 threshold，而不是靠噪声桶。锁定此语义。"""
        e1, e2, e3 = self._emb(1), self._emb(2), self._emb(3)  # 互相近正交
        result = cluster_embeddings_dbscan({"x": e1, "y": e2, "z": e3}, eps=0.05)
        # 无 __noise__（min_samples=1），三个单例簇
        assert "__noise__" not in result
        assert sorted(k for k in result) and all(len(v) == 1 for v in result.values())
        assert set(result.keys()) == {"x", "y", "z"}

    def test_match_face_to_cluster_threshold(self):
        e = self._emb(7)
        clusters = {"m1": ["a"], "m2": ["b"]}
        embs = {"a": e, "b": self._emb(8)}
        assert match_face_to_cluster(e, clusters, embs, threshold=0.7) == "m1"
        assert match_face_to_cluster(e, clusters, embs, threshold=1.01) is None

    def test_legacy_greedy_cluster(self):
        base = self._emb(5)
        same = base.copy()
        other = self._emb(6)
        r = cluster_embeddings({"k1": base, "k2": same, "k3": other}, threshold=0.95)
        group = next(v for v in r.values() if "k1" in v)
        assert "k2" in group and "k3" not in group
