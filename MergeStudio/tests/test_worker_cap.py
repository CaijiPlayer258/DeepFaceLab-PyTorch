"""routes_export.worker 封顶逻辑的独立单元测试（不打 FastAPI）。

封顶代码在 export_start 的 _run() 内联，无法直接调用；
这里把同一段数学抽出来对拍，防止未来改动漂移（P0-6 回归锚）。
"""
import os


def _apply_cap(requested: int, cpu_count: int, env: str = "") -> int:
    """复刻 routes_export.py 的 worker 计算链。"""
    nw = requested if requested > 0 else max(1, cpu_count // 2)
    env_nw = env.strip()
    cap = int(env_nw) if env_nw.isdigit() and int(env_nw) > 0 else 4
    return min(nw, cap)


class TestWorkerCap:

    def test_default_cpu_halving_capped(self):
        assert _apply_cap(0, cpu_count=16) == 4
        assert _apply_cap(0, cpu_count=8) == 4
        assert _apply_cap(0, cpu_count=4) == 2

    def test_explicit_big_capped(self):
        assert _apply_cap(99, cpu_count=32) == 4

    def test_explicit_small_respected(self):
        assert _apply_cap(2, cpu_count=32) == 2

    def test_env_override_raises_cap(self):
        assert _apply_cap(99, cpu_count=32, env="6") == 6

    def test_env_invalid_ignored(self):
        assert _apply_cap(0, cpu_count=32, env="abc") == 4
        assert _apply_cap(0, cpu_count=32, env="-3") == 4
        assert _apply_cap(0, cpu_count=32, env="0") == 4
