
import json, statistics
from pathlib import Path
D = Path(__file__).resolve().parents[4] / "_enc4k_bench"   # 仓库根的 scratch 素材/产物目录（不进仓）
R = json.load(open(D/"results.json")); Q = json.load(open(D/"quality.json"))
C21 = json.load(open(D/"results_cq21.json"))
def med(rows, key):
    v=[r[key] for r in rows if r[key] is not None]
    return statistics.median(v) if v else None
for tag in ("4k","hd","easy4k"):
    rows = R[tag]
    print("==", tag)
    arms = []
    for r in rows:
        if r["arm"] not in arms: arms.append(r["arm"])
    for a in arms:
        rs=[r for r in rows if r["arm"]==a]
        a_key = a.split()[0]
        q = Q.get(tag,{}).get(a_key,{})
        print("  %-22s fps %6.1f  ms/f %5.1f  写 %5.1f  cores %5.2f+%.2f  sm %s enc %s  MB %6.2f  %5.1f Mbps  md5same=%s" % (
            a, med(rs,"fps"), med(rs,"ms_frame"), med(rs,"write_ms"), med(rs,"cores_ffmpeg"),
            med(rs,"cores_writer"), med(rs,"gpu_sm"), med(rs,"gpu_enc"), med(rs,"mb"), med(rs,"mbps"),
            len({r["md5"] for r in rs})==1))
        if q: print("      画质 vs x264: 平均差 %.3f LSB  p99 %.3f  >8LSB %.2f%%  PSNR %.2f dB  SSIM %.5f  | vs 原始帧 PSNR %.2f" % (
            q["mean_diff"], q["p99"], q["gt8_pct"], q["psnr_vs_x264"], q["ssim_vs_x264"], q["psnr_vs_src"]))
print("== CQ21 (4k)")
for k,v in C21.items():
    print("  %-22s fps %6.1f MB %6.2f  %.1f Mbps md5same=%s" % (k, med(v,"fps"), med(v,"mb"), med(v,"mbps"), len({x["md5"] for x in v})==1))
