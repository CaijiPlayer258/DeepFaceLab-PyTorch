"""
DFL PNG 脸图解析/写入器（dFLd 私有块）——纯 Python 实现，替代老 MVE 的
pyarmor 加密 DFLPNG.pyd，使深变（DFL-PyTorch）能直接读老工作流的 PNG aligned 脸集。

块布局（与老 DFLPNG.pyd 实测一致）：
  PNG 签名 + 标准块（IHDR/IDAT/IEND...），其中在 IDAT 前插入自造辅助块：
      len(4B BE) + b'dFLd' + payload + crc32(type+data)(4B BE)
  payload = 与 DFLJPG APP15 完全相同的 pickle 字典
  （face_type / landmarks / source_filename / source_rect / source_landmarks /
    image_to_face_mat / seg_ie_polys / xseg_mask ...）

所有磁盘读写用 np.fromfile/ndarray.tofile，兼容中文路径。
"""
import pickle
import struct
import traceback
import warnings
import zlib

import cv2
import numpy as np

from core.imagelib import SegIEPolys
from core.interact import interact as io
from core.pathex import safe_pickle_load as _safe_pickle_load
from facelib import FaceType

import os


_PNG_SIG = b'\x89PNG\r\n\x1a\n'
_DFL_TYPE = b'dFLd'


def _read_file_bytes(path) -> bytes:
    buf = np.fromfile(str(path), dtype=np.uint8)
    return buf.tobytes()


def _write_file_bytes(path, data: bytes):
    np.frombuffer(data, dtype=np.uint8).tofile(str(path))


def _chunk_bytes(ctype: bytes, cdata: bytes) -> bytes:
    return (struct.pack('>I', len(cdata)) + ctype + cdata +
            struct.pack('>I', zlib.crc32(ctype + cdata) & 0xFFFFFFFF))


class DFLPNG(object):
    def __init__(self, filename):
        self.filename = filename
        self.shape = None
        self.img = None
        self.chunks = []      # [{'type': b'IHDR', 'data': bytes}, ...] 含 IEND
        self.dfl_dict = {}

    # ── 解析 ─────────────────────────────────────────────────────

    @staticmethod
    def load_raw(filename, loader_func=None):
        if loader_func is not None:
            data = loader_func(filename)
        else:
            data = _read_file_bytes(filename)

        if data[:8] != _PNG_SIG:
            raise ValueError(f'No Valid PNG info in {filename}')

        inst = DFLPNG(filename)
        chunks = []
        pos = 8
        while pos + 12 <= len(data):
            ln, = struct.unpack('>I', data[pos:pos + 4])
            ctype = data[pos + 4:pos + 8]
            cdata = data[pos + 8:pos + 8 + ln]
            if len(cdata) < ln:
                raise ValueError(f'truncated chunk {ctype} in {filename}')
            chunks.append({'type': ctype, 'data': cdata})
            pos += 12 + ln
            if ctype == b'IEND':
                break
        inst.chunks = chunks
        return inst

    @staticmethod
    def load(filename, loader_func=None):
        try:
            inst = DFLPNG.load_raw(filename, loader_func=loader_func)
            inst.dfl_dict = {}
            for chunk in inst.chunks:
                if chunk['type'] == b'IHDR':
                    w, h = struct.unpack('>II', chunk['data'][:8])
                    inst.shape = (h, w, 3)
                elif chunk['type'] == _DFL_TYPE:
                    if isinstance(chunk['data'], bytes):
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            inst.dfl_dict = _safe_pickle_load(chunk['data'])
            return inst
        except Exception as e:
            io.log_err(f'Exception occured while DFLPNG.load : {traceback.format_exc()}')
            return None

    # ── 写回 ─────────────────────────────────────────────────────

    def has_data(self):
        return len(self.dfl_dict.keys()) != 0

    def save(self):
        try:
            _write_file_bytes(self.filename, self.dump())
        except:
            raise Exception(f'cannot save {self.filename}')

    def dump(self):
        dict_data = self.dfl_dict

        # Remove None keys
        for key in list(dict_data.keys()):
            if dict_data[key] is None:
                dict_data.pop(key)

        # 与 DFLJPG.dump 相同的 numpy 归一化（xseg_mask 保留 ndarray）
        xseg_val = dict_data.get('xseg_mask')

        def _unumpy(obj):
            if isinstance(obj, np.ndarray):
                if obj is xseg_val:
                    return obj
                if obj.dtype.kind in ('u', 'b'):
                    return bytes(obj)
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer, np.bool_)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _unumpy(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_unumpy(i) for i in obj]
            return obj

        dict_data = _unumpy(dict_data)

        pickle_data = pickle.dumps(dict_data, protocol=2)
        pickle_data = pickle_data.replace(b'numpy._core', b'numpy.core')

        # 重建块序列：去掉旧 dFLd，在第一个 IDAT 前插入新 dFLd
        out = [b'', ]  # 占位：签名
        inserted = False
        for chunk in self.chunks:
            if chunk['type'] == _DFL_TYPE:
                continue
            if not inserted and chunk['type'] == b'IDAT':
                out.append(_chunk_bytes(_DFL_TYPE, pickle_data))
                inserted = True
            out.append(_chunk_bytes(chunk['type'], chunk['data']))
        if not inserted:
            # 没有 IDAT（异常文件）：塞到 IEND 前
            body = b''.join(out[1:])
            idx = body.rfind(struct.pack('>I', 0) + b'IEND')
            if idx >= 0:
                body = body[:idx] + _chunk_bytes(_DFL_TYPE, pickle_data) + body[idx:]
            return _PNG_SIG + body
        return _PNG_SIG + b''.join(out[1:])

    # ── 读取（与 DFLJPG 方法面完全一致）───────────────────────────

    def get_img(self):
        if self.img is None:
            try:
                data = _read_file_bytes(self.filename)
                self.img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            except:
                self.img = None
        return self.img

    def get_shape(self):
        if self.shape is None:
            img = self.get_img()
            if img is not None:
                self.shape = img.shape
        return self.shape

    def get_height(self):
        if self.shape is not None:
            return self.shape[0]
        img = self.get_img()
        if img is not None:
            return img.shape[0]
        return 0

    def get_dict(self):
        return self.dfl_dict

    def set_dict(self, dict_data=None):
        self.dfl_dict = dict_data

    def get_face_type(self):            return self.dfl_dict.get('face_type', FaceType.toString(FaceType.FULL))
    def set_face_type(self, face_type): self.dfl_dict['face_type'] = face_type

    def get_landmarks(self):            return np.array(self.dfl_dict['landmarks'])
    def set_landmarks(self, landmarks): self.dfl_dict['landmarks'] = landmarks

    def get_eyebrows_expand_mod(self):                      return self.dfl_dict.get('eyebrows_expand_mod', 1.0)
    def set_eyebrows_expand_mod(self, eyebrows_expand_mod): self.dfl_dict['eyebrows_expand_mod'] = eyebrows_expand_mod

    def get_source_filename(self):                  return self.dfl_dict.get('source_filename', None)
    def set_source_filename(self, source_filename): self.dfl_dict['source_filename'] = source_filename

    def get_source_rect(self):              return self.dfl_dict.get('source_rect', None)
    def set_source_rect(self, source_rect): self.dfl_dict['source_rect'] = source_rect

    def get_source_landmarks(self):
        val = self.dfl_dict.get('source_landmarks', None)
        if val is None:
            return None
        return np.array(val)

    def set_source_landmarks(self, source_landmarks): self.dfl_dict['source_landmarks'] = source_landmarks

    def get_image_to_face_mat(self):
        mat = self.dfl_dict.get('image_to_face_mat', None)
        if mat is not None:
            return np.array(mat)
        return None

    def set_image_to_face_mat(self, image_to_face_mat): self.dfl_dict['image_to_face_mat'] = image_to_face_mat

    def has_seg_ie_polys(self):
        return self.dfl_dict.get('seg_ie_polys', None) is not None

    def get_seg_ie_polys(self):
        d = self.dfl_dict.get('seg_ie_polys', None)
        if d is not None:
            d = SegIEPolys.load(d)
        else:
            d = SegIEPolys()
        return d

    def set_seg_ie_polys(self, seg_ie_polys):
        if seg_ie_polys is not None:
            if not isinstance(seg_ie_polys, SegIEPolys):
                raise ValueError('seg_ie_polys should be instance of SegIEPolys')

            if seg_ie_polys.has_polys():
                seg_ie_polys = seg_ie_polys.dump()
            else:
                seg_ie_polys = None

        self.dfl_dict['seg_ie_polys'] = seg_ie_polys

    def has_xseg_mask(self):
        return self.dfl_dict.get('xseg_mask', None) is not None

    def get_xseg_mask_compressed(self):
        return self.dfl_dict.get('xseg_mask', None)

    def get_xseg_mask(self):
        mask_buf = self.dfl_dict.get('xseg_mask', None)
        if mask_buf is None:
            return None
        if isinstance(mask_buf, bytes):
            mask_buf = np.frombuffer(mask_buf, dtype=np.uint8)

        img = cv2.imdecode(mask_buf, cv2.IMREAD_UNCHANGED)
        if len(img.shape) == 2:
            img = img[..., None]

        return img.astype(np.float32) / 255.0

    def set_xseg_mask(self, mask_a):
        if mask_a is None:
            self.dfl_dict['xseg_mask'] = None
            return

        from core import imagelib
        mask_a = imagelib.normalize_channels(mask_a, 1)
        img_data = np.clip(mask_a * 255, 0, 255).astype(np.uint8)

        data_max_len = 50000

        ret, buf = cv2.imencode('.png', img_data)

        if not ret or len(buf) > data_max_len:
            for jpeg_quality in range(100, -1, -1):
                ret, buf = cv2.imencode('.jpg', img_data, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
                if ret and len(buf) <= data_max_len:
                    break

        if not ret:
            raise Exception('set_xseg_mask: unable to generate image data for set_xseg_mask')

        self.dfl_dict['xseg_mask'] = buf
