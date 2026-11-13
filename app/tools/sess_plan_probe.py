
import sys, os
sys.path.insert(0, r"D:\MyProject\DeepFaceLab-PyTorch-4.1\app")
os.environ["DFL_WORKSPACE"] = r"C:\Users\30764\AppData\Local\Temp\dfl_derive_probe\ws"
from server import _sess
p = _sess.probe(vram_free_mb=3351, vram_total_mb=8150)
print("probe", p)
pl = _sess.derive("detect", p, 16.1, 57.4, mem_per_gpu_mb=300.0, mem_per_cpu_mb=260.0)
print("auto:", _sess.plan_line(pl))
pl2 = _sess.derive("detect", p, 16.1, 57.4, mem_per_gpu_mb=300.0, mem_per_cpu_mb=260.0,
                   over={"gpu": None, "cpu": 0})
print("g auto c0:", _sess.plan_line(pl2))
p3 = dict(p); p3.update(_sess.level_cfg("low"))
print("low:", _sess.plan_line(_sess.derive("detect", p3, 16.1, 57.4, 300.0, 260.0)))
p4 = dict(p); p4.update(_sess.level_cfg("high"))
print("high:", _sess.plan_line(_sess.derive("detect", p4, 16.1, 57.4, 300.0, 260.0)))
p5 = dict(p); p5.update(_sess.level_cfg("nogpu"))
print("nogpu:", _sess.plan_line(_sess.derive("detect", p5, None, 57.4, 300.0, 260.0)))
