"""
Face swap merge engine.
Transplanted from merger/MergeMasked.py - preserving exact flow.
"""
from pathlib import Path
import cv2
import numpy as np
import traceback
from facelib import FaceType, LandmarksProcessor
from core import imagelib

_model_input_size = 256  # Overridden by model_loader


def _build_debug_vis(prd_face_bgr, dst_face_bgr, wrk_face_mask_a_0,
                     prd_face_mask_a_0=None, prd_face_dst_mask_a_0=None, output_size=256):
    """Build debug visualization: 2 rows x 3 cols of output_size images."""
    def colorize_mask(mk):
        m_8u = (np.clip(mk, 0, 1) * 255).astype(np.uint8)
        if m_8u.ndim == 2:
            return cv2.applyColorMap(m_8u, cv2.COLORMAP_VIRIDIS)
        return cv2.applyColorMap(m_8u[:, :, 0], cv2.COLORMAP_VIRIDIS) if m_8u.ndim == 3 else m_8u

    vis = np.zeros((output_size * 2 + 30, output_size * 3 + 30, 3), dtype=np.uint8)
    vis[:] = 32
    gap = 10
    os = output_size
    # Row 1: masks (use model/XSeg masks, fallback to wrk)
    mask_items = [
        ('pred', prd_face_mask_a_0 if prd_face_mask_a_0 is not None else wrk_face_mask_a_0),
        ('dst', prd_face_dst_mask_a_0 if prd_face_dst_mask_a_0 is not None else wrk_face_mask_a_0),
        ('prd*dst', wrk_face_mask_a_0),
    ]
    for i, (name, mk) in enumerate(mask_items):
        x = gap + i * (os + gap)
        if mk is not None and hasattr(mk, 'shape') and mk.ndim >= 2 and mk.size > 0:
            mk_2d = mk.squeeze() if mk.ndim > 2 else mk
            if mk_2d.shape[:2] != (os, os):
                mk_2d = cv2.resize(mk_2d, (os, os), interpolation=cv2.INTER_CUBIC)
            if mk_2d.ndim == 2 and mk_2d.size > 0:
                colored = colorize_mask(mk_2d)
                vis[gap:gap + os, x:x + os] = colored
                cv2.putText(vis, name, (x + 2, gap + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    # Row 2: RGB images
    y2 = gap + os + gap
    prd_rgb = cv2.resize(prd_face_bgr, (os, os)) if prd_face_bgr.shape[:2] != (os, os) else prd_face_bgr[:os, :os]
    dst_rgb = cv2.resize(dst_face_bgr, (os, os)) if dst_face_bgr.shape[:2] != (os, os) else dst_face_bgr[:os, :os]
    if wrk_face_mask_a_0 is not None:
        mk_2d = wrk_face_mask_a_0.squeeze()
        if mk_2d.shape[:2] != (os, os):
            mk_2d = cv2.resize(mk_2d, (os, os), interpolation=cv2.INTER_CUBIC)
        mk_3ch = np.stack([mk_2d] * 3, axis=-1)
    else:
        mk_3ch = np.ones((os, os, 3))
    merged = prd_rgb * mk_3ch + dst_rgb * (1 - mk_3ch)
    for j, (name, img) in enumerate([('pred RGB', prd_rgb), ('dst RGB', dst_rgb), ('merged', merged)]):
        x = gap + j * (os + gap)
        vis[y2:y2 + os, x:x + os] = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        cv2.putText(vis, name, (x + 2, y2 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1)
    return vis


def MergeMaskedFace(frame_img, face_landmarks, cfg, predictor_func=None,
                    face_enhancer_func=None, xseg_256_extract_func=None):
    """
    Merge a single face. Matches original MergeMasked.py flow exactly.
    Everything works in 0-1 float range. Predictor returns [face, prd_mask, dst_mask].
    """
    try:
        if isinstance(cfg.face_type, str):
            ft_map = {'half_face': FaceType.HALF, 'midfull_face': FaceType.MID_FULL,
                      'full_face': FaceType.FULL, 'whole_face': FaceType.WHOLE_FACE, 'head': FaceType.HEAD}
            cfg.face_type = ft_map.get(cfg.face_type, FaceType.WHOLE_FACE)

        input_size = _model_input_size
        output_size = input_size
        if cfg.super_resolution_power != 0 and face_enhancer_func is not None:
            output_size *= 4  # 仅在 face_enhancer 可用时才放大（否则预测脸与蒙版尺寸不匹配导致黑脸）

        # Normalize frame to 0-1
        img_bgr = frame_img.astype(np.float32) / 255.0
        img_size = (frame_img.shape[1], frame_img.shape[0])

        # Hull mask on full frame
        img_face_mask_a = LandmarksProcessor.get_image_hull_mask(img_bgr.shape, face_landmarks)

        # Get transform mats
        face_mat = LandmarksProcessor.get_transform_mat(face_landmarks, output_size, face_type=cfg.face_type)
        scale = 1.0 + 0.01 * cfg.output_face_scale
        face_output_mat = LandmarksProcessor.get_transform_mat(face_landmarks, output_size, face_type=cfg.face_type, scale=scale)

        # Extract face + mask
        dst_face_bgr = cv2.warpAffine(img_bgr, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
        dst_face_bgr = np.clip(dst_face_bgr, 0, 1)

        dst_face_mask_a_0 = cv2.warpAffine(img_face_mask_a, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)
        dst_face_mask_a_0 = np.clip(dst_face_mask_a_0, 0, 1)

        # Predict
        predictor_input_bgr = cv2.resize(dst_face_bgr, (input_size, input_size))
        if predictor_func is not None:
            result = predictor_func(predictor_input_bgr)
            if isinstance(result, tuple):
                predicted = list(result)
            else:
                predicted = [result, None, None]
        else:
            predicted = [dst_face_bgr, None, None]

        prd_face_bgr = np.clip(predicted[0], 0, 1.0) if predicted[0] is not None else dst_face_bgr
        prd_face_mask_a_0 = np.clip(predicted[1], 0, 1.0) if len(predicted) > 1 and predicted[1] is not None else None
        prd_face_dst_mask_a_0 = np.clip(predicted[2], 0, 1.0) if len(predicted) > 2 and predicted[2] is not None else None

        # Super resolution
        if cfg.super_resolution_power != 0 and face_enhancer_func is not None:
            enhanced = face_enhancer_func(prd_face_bgr, is_tanh=True, preserve_size=False)
            mod = cfg.super_resolution_power / 100.0
            prd_face_bgr = cv2.resize(prd_face_bgr, (output_size, output_size)) * (1.0 - mod) + enhanced * mod
            prd_face_bgr = np.clip(prd_face_bgr, 0, 1)
            if prd_face_mask_a_0 is not None:
                prd_face_mask_a_0 = cv2.resize(prd_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            if prd_face_dst_mask_a_0 is not None:
                prd_face_dst_mask_a_0 = cv2.resize(prd_face_dst_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

        # When seg_mode is XSeg/XSegLite, replace model masks with XSeg results
        seg = getattr(cfg, 'seg_mode', 'model')
        if seg in ('xseg', 'xseglite') and xseg_256_extract_func is not None:
            prd_face_xseg = cv2.resize(prd_face_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
            xseg_result_prd = xseg_256_extract_func(prd_face_xseg)
            X_prd = cv2.resize(xseg_result_prd, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            xseg_mat = LandmarksProcessor.get_transform_mat(face_landmarks, 256, face_type=cfg.face_type)
            dst_xseg = cv2.warpAffine(img_bgr, xseg_mat, (256, 256), flags=cv2.INTER_CUBIC)
            xseg_result_dst = xseg_256_extract_func(dst_xseg)
            X_dst = cv2.resize(xseg_result_dst, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            #print(f"[Mask] XSeg({seg}): prd mean={X_prd.mean():.3f} max={X_prd.max():.3f}, dst mean={X_dst.mean():.3f} max={X_dst.max():.3f}")
            #print(f"[Mask] XSeg dst_xseg: shape={dst_xseg.shape} range=[{dst_xseg.min():.3f},{dst_xseg.max():.3f}] mean={dst_xseg.mean():.3f}")
            #print(f"[Mask] XSeg result: prd range=[{xseg_result_prd.min():.3f},{xseg_result_prd.max():.3f}] mean={xseg_result_prd.mean():.3f}, "
            #      f"dst range=[{xseg_result_dst.min():.3f},{xseg_result_dst.max():.3f}] mean={xseg_result_dst.mean():.3f}")
            # Override model masks with XSeg masks
            prd_face_mask_a_0 = np.clip(X_prd, 0, 1)
            prd_face_dst_mask_a_0 = np.clip(X_dst, 0, 1)
        elif seg == 'model' and prd_face_mask_a_0 is not None and prd_face_dst_mask_a_0 is not None:
            pass  # DFM masks already set above

        # Build working mask
        if cfg.mask_mode == 0:  # full
            wrk_face_mask_a_0 = np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 1:  # dst
            wrk_face_mask_a_0 = cv2.resize(dst_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        elif cfg.mask_mode == 2:  # learned-prd
            wrk_face_mask_a_0 = prd_face_mask_a_0 if prd_face_mask_a_0 is not None else np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 3:  # learned-dst
            wrk_face_mask_a_0 = prd_face_dst_mask_a_0 if prd_face_dst_mask_a_0 is not None else np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 4:  # learned-prd*learned-dst
            if prd_face_mask_a_0 is not None and prd_face_dst_mask_a_0 is not None:
                wrk_face_mask_a_0 = prd_face_mask_a_0 * prd_face_dst_mask_a_0
            else:
                wrk_face_mask_a_0 = np.ones_like(dst_face_mask_a_0)
        elif cfg.mask_mode == 5:  # learned-prd+learned-dst
            mask = np.zeros_like(dst_face_mask_a_0)
            if prd_face_mask_a_0 is not None: mask += prd_face_mask_a_0
            if prd_face_dst_mask_a_0 is not None: mask += prd_face_dst_mask_a_0
            wrk_face_mask_a_0 = np.clip(mask, 0, 1)
        elif cfg.mask_mode >= 6:  # XSeg modes (old style, kept for compat)
            if cfg.mask_mode in (6, 8, 9) and xseg_256_extract_func is not None:
                prd_face_xseg = cv2.resize(prd_face_bgr, (256, 256), interpolation=cv2.INTER_CUBIC)
                prd_xseg_mask = xseg_256_extract_func(prd_face_xseg)
                X_prd = cv2.resize(prd_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            else:
                X_prd = None
            if cfg.mask_mode >= 7 and xseg_256_extract_func is not None:
                xseg_mat = LandmarksProcessor.get_transform_mat(face_landmarks, 256, face_type=cfg.face_type)
                dst_xseg = cv2.warpAffine(img_bgr, xseg_mat, (256, 256), flags=cv2.INTER_CUBIC)
                dst_xseg_mask = xseg_256_extract_func(dst_xseg)
                X_dst = cv2.resize(dst_xseg_mask, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            else:
                X_dst = None

            if cfg.mask_mode == 6:
                wrk_face_mask_a_0 = X_prd if X_prd is not None else np.ones_like(dst_face_mask_a_0)
            elif cfg.mask_mode == 7:
                wrk_face_mask_a_0 = X_dst if X_dst is not None else np.ones_like(dst_face_mask_a_0)
            elif cfg.mask_mode == 8:
                wrk_face_mask_a_0 = X_prd * X_dst if (X_prd is not None and X_dst is not None) else np.ones_like(dst_face_mask_a_0)
            elif cfg.mask_mode == 9:
                mask = np.ones_like(dst_face_mask_a_0)
                if prd_face_mask_a_0 is not None: mask *= prd_face_mask_a_0
                if prd_face_dst_mask_a_0 is not None: mask *= prd_face_dst_mask_a_0
                if X_prd is not None: mask *= X_prd
                if X_dst is not None: mask *= X_dst
                wrk_face_mask_a_0 = mask
        else:
            wrk_face_mask_a_0 = dst_face_mask_a_0

        #print(f"[Mask] Final mask (seg={seg}, mode={cfg.mask_mode}): mean={wrk_face_mask_a_0.mean():.3f}, max={wrk_face_mask_a_0.max():.3f}")
        wrk_face_mask_a_0[wrk_face_mask_a_0 < (1.0 / 255.0)] = 0.0

        # Process mask (erode/blur) in padded space
        if 'raw' not in cfg.mode:
            wrk_face_mask_a_0 = np.pad(wrk_face_mask_a_0, input_size)
            ero = cfg.erode_mask_modifier
            blur = cfg.blur_mask_modifier
            if ero > 0:
                wrk_face_mask_a_0 = cv2.erode(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ero, ero)), iterations=1)
            elif ero < 0:
                wrk_face_mask_a_0 = cv2.dilate(wrk_face_mask_a_0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (-ero, -ero)), iterations=1)
            clip_size = input_size + blur // 2
            wrk_face_mask_a_0[:clip_size, :] = 0
            wrk_face_mask_a_0[-clip_size:, :] = 0
            wrk_face_mask_a_0[:, :clip_size] = 0
            wrk_face_mask_a_0[:, -clip_size:] = 0
            if blur > 0:
                blur = blur + (1 - blur % 2)
                wrk_face_mask_a_0 = cv2.GaussianBlur(wrk_face_mask_a_0, (blur, blur), 0)
            wrk_face_mask_a_0 = wrk_face_mask_a_0[input_size:-input_size, input_size:-input_size]
            wrk_face_mask_a_0 = np.clip(wrk_face_mask_a_0, 0, 1)

        # Warp mask to full frame and resize to output_size
        img_face_mask_full = cv2.warpAffine(
            wrk_face_mask_a_0, face_output_mat, img_size,
            np.zeros(img_bgr.shape[:2], dtype=np.float32),
            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC
        )[..., None]
        img_face_mask_full = np.clip(img_face_mask_full, 0, 1)
        img_face_mask_full[img_face_mask_full < (1.0 / 255.0)] = 0.0

        if wrk_face_mask_a_0.shape[0] != output_size:
            wrk_face_mask_a_0 = cv2.resize(wrk_face_mask_a_0, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
        wrk_face_mask_a = wrk_face_mask_a_0[..., None]

        if 'raw' in cfg.mode:
            if cfg.mode == 'raw-rgb':
                out_face = cv2.warpAffine(prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                out_mask = cv2.warpAffine(np.ones_like(prd_face_bgr), face_output_mat, img_size, np.empty_like(img_bgr), cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                out_img = img_bgr * (1 - out_mask) + out_face * out_mask
                out_merging_mask_a = img_face_mask_full
            elif cfg.mode == 'raw-predict':
                out_img = prd_face_bgr
                out_merging_mask_a = wrk_face_mask_a
            out_img = np.clip(out_img, 0, 1)
        else:
            # Standard mode
            maxregion = np.argwhere(img_face_mask_full >= 0.1)
            if maxregion.size != 0:
                miny, minx = maxregion.min(axis=0)[:2]
                maxy, maxx = maxregion.max(axis=0)[:2]
                if min(maxy - miny, maxx - minx) >= 4:
                    mask_area = wrk_face_mask_a.copy()
                    mask_area[mask_area > 0] = 1.0

                    # Color transfer functions (defined here for scoping)
                    ct_functions = {
                            1: lambda s, t: imagelib.reinhard_color_transfer(s, t, target_mask=mask_area, source_mask=mask_area),
                            2: lambda s, t: imagelib.linear_color_transfer(s, t),
                            3: lambda s, t: imagelib.color_transfer_mkl(s, t),
                            4: lambda s, t: imagelib.color_transfer_mkl(s * mask_area, t * mask_area),
                            5: lambda s, t: imagelib.color_transfer_idt(s, t),
                            6: lambda s, t: imagelib.color_transfer_idt(s * mask_area, t * mask_area),
                            7: lambda s, t: imagelib.color_transfer_sot(s * mask_area, t * mask_area, steps=10, batch_size=30),
                            8: lambda s, t: imagelib.color_transfer_mix(s * mask_area, t * mask_area),
                            # 9/10 与训练预览侧对齐（config.ctm_dict 同步）
                            9: lambda s, t: imagelib.match_tone_np(s, t, src_mask=mask_area, trg_mask=mask_area),
                            10: lambda s, t: imagelib.color_transfer_lut(s, t),
                    }  # ct_functions
                    fn = ct_functions.get(cfg.color_transfer_mode)
                    if fn:
                        try:
                            prd_face_bgr = np.clip(fn(prd_face_bgr, dst_face_bgr), 0, 1)
                        except Exception:
                            pass

                    # Hist-match
                    if cfg.mode == 'hist-match':
                        hist_mask = np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)
                        if cfg.masked_hist_match:
                            hist_mask *= mask_area
                        white = (1.0 - hist_mask) * np.ones(prd_face_bgr.shape[:2] + (1,), dtype=np.float32)
                        hm1 = prd_face_bgr * hist_mask + white
                        hm1[hm1 > 1.0] = 1.0
                        hm2 = dst_face_bgr * hist_mask + white
                        hm2[hm2 > 1.0] = 1.0
                        prd_face_bgr = imagelib.color_hist_match(hm1, hm2, cfg.hist_match_threshold).astype(np.float32)

                    # Seamless mask
                    img_face_seamless_mask = None
                    if 'seamless' in cfg.mode:
                        for i in range(1, 10):
                            a = img_face_mask_full > i / 10.0
                            if len(np.argwhere(a)) == 0:
                                continue
                            img_face_seamless_mask = img_face_mask_full.copy()
                            img_face_seamless_mask[a] = 1.0
                            img_face_seamless_mask[img_face_seamless_mask <= i / 10.0] = 0.0
                            break

                    # Warp predicted face to full frame
                    out_img = cv2.warpAffine(prd_face_bgr, face_output_mat, img_size, np.empty_like(img_bgr),
                                            flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                    out_img = np.clip(out_img, 0, 1)

                    # Seamless clone
                    if 'seamless' in cfg.mode and img_face_seamless_mask is not None:
                        try:
                            l, t, w, h = cv2.boundingRect((img_face_seamless_mask * 255).astype(np.uint8))
                            center = (int(l + w / 2), int(t + h / 2))
                            out_img = cv2.seamlessClone(
                                (out_img * 255).astype(np.uint8),
                                frame_img.astype(np.uint8),
                                (img_face_seamless_mask * 255).astype(np.uint8),
                                center, cv2.NORMAL_CLONE
                            ).astype(np.float32) / 255.0
                        except Exception:
                            pass

                    # Composite with mask
                    out_img = img_bgr * (1 - img_face_mask_full) + out_img * img_face_mask_full

                    # Post-processing (on warped-back face)
                    mp = cfg.motion_blur_power / 100.0
                    needs_pp = ('seamless' in cfg.mode and cfg.color_transfer_mode != 0) or \
                               cfg.mode == 'seamless-hist-match' or mp != 0 or \
                               cfg.blursharpen_amount != 0 or cfg.image_denoise_power != 0 or \
                               cfg.bicubic_degrade_power != 0

                    if needs_pp:
                        out_face_bgr_pp = cv2.warpAffine(out_img, face_mat, (output_size, output_size), flags=cv2.INTER_CUBIC)

                        if 'seamless' in cfg.mode and cfg.color_transfer_mode != 0:
                            try:
                                fn_pp = ct_functions.get(cfg.color_transfer_mode)
                                if fn_pp:
                                    out_face_bgr_pp = np.clip(fn_pp(out_face_bgr_pp, dst_face_bgr), 0, 1)
                            except Exception:
                                pass
                                try:
                                    out_face_bgr_pp = np.clip(fn_pp(out_face_bgr_pp, dst_face_bgr), 0, 1)
                                except Exception:
                                    pass

                        if cfg.mode == 'seamless-hist-match':
                            out_face_bgr_pp = imagelib.color_hist_match(out_face_bgr_pp, dst_face_bgr, cfg.hist_match_threshold)

                        if mp != 0:
                            k = int(mp * 100)
                            if k >= 1:
                                k = np.clip(k + 1, 2, 50)
                                kernel = np.zeros((k, k))
                                kernel[:, k // 2] = 1.0 / k
                                out_face_bgr_pp = cv2.filter2D(out_face_bgr_pp, -1, kernel)

                        if cfg.blursharpen_amount != 0:
                            amt = cfg.blursharpen_amount / 100.0
                            if cfg.sharpen_mode == 1:
                                blurred = cv2.blur(out_face_bgr_pp, (3, 3))
                            else:
                                blurred = cv2.GaussianBlur(out_face_bgr_pp, (0, 0), 1.0)
                            out_face_bgr_pp = cv2.addWeighted(out_face_bgr_pp, 1.0 + amt, blurred, -amt, 0)
                            out_face_bgr_pp = np.clip(out_face_bgr_pp, 0, 1)

                        if cfg.image_denoise_power != 0:
                            n = cfg.image_denoise_power
                            while n > 0:
                                denoised = cv2.medianBlur(frame_img.astype(np.float32) / 255.0, 5)
                                if int(n / 100) != 0:
                                    img_bgr = denoised
                                else:
                                    p = (n % 100) / 100.0
                                    img_bgr = img_bgr * (1.0 - p) + denoised * p
                                n = max(n - 10, 0)

                        if cfg.bicubic_degrade_power != 0:
                            p = 1.0 - cfg.bicubic_degrade_power / 101.0
                            down = cv2.resize(img_bgr, (int(img_size[0] * p), int(img_size[1] * p)), interpolation=cv2.INTER_CUBIC)
                            img_bgr = cv2.resize(down, img_size, interpolation=cv2.INTER_CUBIC)

                        new_out = cv2.warpAffine(out_face_bgr_pp, face_mat, img_size, np.empty_like(img_bgr),
                                                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC)
                        out_img = np.clip(img_bgr * (1 - img_face_mask_full) + new_out * img_face_mask_full, 0, 1)

                    if cfg.color_degrade_power != 0:
                        reduced = imagelib.reduce_colors(out_img, 256)
                        if cfg.color_degrade_power == 100:
                            out_img = reduced
                        else:
                            alpha = cfg.color_degrade_power / 100.0
                            out_img = out_img * (1.0 - alpha) + reduced * alpha

            out_merging_mask_a = img_face_mask_full

        if 'out_img' not in dir() or out_img is None:
            # P1-8: mask_mode=0 with degenerate maxregion leaves out_img unbound;
            # fall back to the original frame instead of raising NameError.
            out_img = img_bgr.copy()

        # Debug visualization (replaces normal output)
        if cfg.mode == 'debug' or getattr(cfg, 'show_debug', False):
            vis = _build_debug_vis(prd_face_bgr, dst_face_bgr, wrk_face_mask_a_0,
                                   prd_face_mask_a_0, prd_face_dst_mask_a_0, output_size)
            return vis.astype(np.uint8), np.zeros((1, 1), dtype=np.uint8)

        final = np.clip(out_img * 255, 0, 255).astype(np.uint8)
        final_mask = np.clip(out_merging_mask_a * 255 if out_merging_mask_a is not None else 0, 0, 255).astype(np.uint8)
        return final, final_mask.squeeze()

    except Exception as e:
        traceback.print_exc()
        return frame_img, np.zeros(frame_img.shape[:2], dtype=np.uint8)
