# dfm_test_merge.py — 用任意 .dfm(ONNX) 对 data_dst 帧做测试合成（驱动 MergeStudio.core.merger）
# 用法: python tools/dfm_test_merge.py <dfm路径> <帧目录> <aligned目录> <输出目录> [erode=20] [blur=80]
import sys, os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import onnxruntime as ort

from MergeStudio.core.merger import MergeMaskedFace
import MergeStudio.core.merger as merger_mod
from MergeStudio.core.config import MergerConfigMasked
from facelib import FaceType
from DFLIMG import DFLPNG

IMG_EXT = {'.png', '.jpg', '.jpeg', '.webp'}


def _norm_face(arr, res):
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[..., None]
    if a.ndim == 3 and a.shape[0] == a.shape[1] == res and a.shape[-1] not in (1, 3) and a.shape[0] in (1, 3):
        a = np.transpose(a, (1, 2, 0))  # NCHW -> NHWC
    return np.clip(a.astype(np.float32), 0, 1)


def _norm_mask(arr, res):
    a = np.asarray(arr).astype(np.float32)
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]  # (H,W,1) -> (H,W)，merger 内部 pad/erode 要求 2D
    return np.clip(a, 0, 1)


def make_predictor(dfm_path, morph_value=1.0):
    try:
        from MergeStudio.core.export_pipeline import _setup_dll_paths
        _setup_dll_paths()  # 补 CUDA/cudnn DLL 路径，否则 onnxruntime 退回 CPU
    except Exception:
        pass
    sess = ort.InferenceSession(str(dfm_path),
                                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    shape = inp.shape
    if len(shape) == 4:
        res = int(shape[1]) if shape[1] == shape[2] else int(shape[2])
    else:
        raise RuntimeError(f'不支持的输入形状: {shape}')
    provider = sess.get_providers()[0]
    extra_names = [i.name for i in sess.get_inputs()[1:]]
    is_amp = any('morph' in n for n in extra_names)
    print(f'[dfm] {Path(dfm_path).name}  input={inp.name} res={res}x{res} provider={provider}'
          + (f'  AMP(+{extra_names}) morph={morph_value}' if is_amp else ''), flush=True)
    merger_mod._model_input_size = res  # 关键：覆盖全局合成尺寸

    def predictor(face_bgr):
        x = face_bgr[None, ...].astype(np.float32)
        if x.shape[1] != res:
            x = np.transpose(x, (0, 3, 1, 2))  # 兜底 NCHW 模型
        feed = {inp.name: x}
        for n in extra_names:
            if 'morph' in n:
                feed[n] = np.array([float(morph_value)], dtype=np.float32)
        outs = sess.run(None, feed)
        face = np.clip(np.asarray(outs[1][0], dtype=np.float32), 0, 1)
        if face.ndim == 2:
            face = face[..., None]
        m_prd = _norm_mask(outs[0][0], res)
        m_dst = _norm_mask(outs[2][0], res)
        return face, m_prd, m_dst

    def _self_check():
        # 预测有效性自检：黑图/噪声图的输出必须与输入不同，否则该 dfm 实际没生效
        x = np.random.RandomState(0).rand(res, res, 3).astype(np.float32)
        y, _, _ = predictor(x)
        d = float(np.abs(y.reshape(y.shape[0], -1) if y.ndim == 3 else y - x).max()) if y.shape == x.shape else 1.0
        if y.shape == x.shape:
            d = float(np.abs(y - x).max())
        if d < (1.0 / 255.0):
            print(f'[FAIL-SELF-CHECK] 预测输出与输入完全相同 —— 该 dfm 推理未生效（检查输入名/morph/feed）', flush=True)
            return False
        print(f'[self-check] 输出与输入最大差 {d:.3f} — 推理有效', flush=True)
        return True

    return predictor, res, _self_check


def main():
    dfm_path = Path(sys.argv[1])
    frames_dir = Path(sys.argv[2])
    aligned_dir = Path(sys.argv[3])
    out_dir = Path(sys.argv[4])
    erode = int(sys.argv[5]) if len(sys.argv) > 5 else 20
    blur = int(sys.argv[6]) if len(sys.argv) > 6 else 80
    morph = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0

    out_dir.mkdir(parents=True, exist_ok=True)
    predictor, res, self_check = make_predictor(dfm_path, morph)
    if not self_check():
        sys.exit(2)

    cfg = MergerConfigMasked(face_type=FaceType.FULL, mode='overlay', mask_mode=4,
                             seg_mode='model', erode_mask_modifier=erode,
                             blur_mask_modifier=blur, color_transfer_mode=1)

    frames = sorted(p for p in frames_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXT)
    print(f'[frames] {len(frames)} 张', flush=True)

    ok = skipped = 0
    for i, f in enumerate(frames, 1):
        lmk_file = aligned_dir / f'{f.stem}_0.npy'
        if lmk_file.exists():
            lmk = np.load(lmk_file)
        else:
            # PNG 脸图直读：从内嵌 dFLd 块取 landmark（需 DFLPNG 支持）
            aligned_png = aligned_dir / f'{f.stem}_0.png'
            dfl = DFLPNG.load(str(aligned_png)) if aligned_png.exists() else None
            lmk = dfl.get_source_landmarks() if dfl is not None and dfl.has_data() else None
        if lmk is None or len(lmk) == 0:
            print(f'[skip] {f.name}: landmark 为空', flush=True)
            skipped += 1
            continue
        img = cv2.imdecode(np.fromfile(str(f), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f'[skip] {f.name}: 读取失败', flush=True)
            skipped += 1
            continue
        try:
            out, _mask_out = MergeMaskedFace(img, lmk, cfg, predictor_func=predictor)
        except Exception as ex:
            print(f'[fail] {f.name}: {ex}', flush=True)
            skipped += 1
            continue
        dst = out_dir / f.name
        if dst.suffix.lower() == '.webp':
            dst = out_dir / f'{f.stem}.png'
        okw, buf = cv2.imencode(dst.suffix.lower(), out)
        if okw:
            buf.tofile(str(dst))
        else:
            print(f'[fail] {f.name}: 编码失败', flush=True)
            skipped += 1
            continue
        ok += 1
        if i % 5 == 0 or i == len(frames):
            print(f'[progress] {i}/{len(frames)} ok={ok}', flush=True)

    print(f'[done] ok={ok} skip/fail={skipped} -> {out_dir}', flush=True)


if __name__ == '__main__':
    main()
