"""
ONNX 模型库 —— 安全导入。

任一模型模块缺失/导入失败时自动跳过（打印警告），
不影响其他模型，也不会导致 `import modelhub.onnx` 或程序启动失败。
可用模型见 `modelhub.onnx.__all__`。
"""
import importlib

# (导出名, 模块路径, 类名)
_MODELS = [
    ('ArcFace',          'modelhub.onnx.ArcFace.ArcFace',                    'ArcFace'),
    ('BlazeFace',        'modelhub.onnx.BlazeFace.BlazeFace',                'BlazeFace'),
    ('CenterFace',       'modelhub.onnx.CenterFace.CenterFace',              'CenterFace'),
    ('FastFaceAlign',    'modelhub.onnx.FastFaceAlign.FastFaceAlign',        'FastFaceAlign'),
    ('DamoFD',           'modelhub.onnx.DamoFD.DamoFD',                      'DamoFD'),
    ('FAN',              'modelhub.onnx.FAN.FAN',                            'FAN'),
    ('FaceEnhancer',     'modelhub.onnx.FaceEnhancer.FaceEnhancer',          'FaceEnhancer'),
    ('FaceMesh',         'modelhub.onnx.FaceMesh.FaceMesh',                  'FaceMesh'),
    ('GenderAge',        'modelhub.onnx.GenderAge.GenderAge',                'GenderAge'),
    ('InsightFace2D106', 'modelhub.onnx.InsightFace2d106.InsightFace2D106',  'InsightFace2D106'),
    ('InsightFace3D68',  'modelhub.onnx.InsightFace3D68.InsightFace3D68',    'InsightFace3D68'),
    ('LIA',              'modelhub.onnx.LIA.LIA',                            'LIA'),
    ('MTCNN',            'modelhub.onnx.MTCNN.MTCNN',                        'MTCNN'),
    ('MogFace',          'modelhub.onnx.MogFace.MogFace',                    'MogFace'),
    ('RetinaFace',       'modelhub.onnx.RetinaFace.RetinaFace',              'RetinaFace'),
    ('S3FD',             'modelhub.onnx.S3FD.S3FD',                          'S3FD'),
    ('TinyMog',          'modelhub.onnx.TinyMog.TinyMog',                    'TinyMog'),
    ('ULFD',             'modelhub.onnx.ULFD.ULFD',                          'ULFD'),
    ('YoloV5Face',       'modelhub.onnx.YoloV5Face.YoloV5Face',              'YoloV5Face'),
    ('YoloV8Face',       'modelhub.onnx.YoloV8Face.YoloV8Face',              'YoloV8Face'),
    ('YoloV11nFace',     'modelhub.onnx.YoloV11nFace.YoloV11nFace',          'YoloV11nFace'),
    ('FaceParser',       'modelhub.onnx.FaceParser.FaceParser',              'FaceParser'),
]

__all__ = []
for _name, _mod, _cls in _MODELS:
    try:
        _m = importlib.import_module(_mod)
        globals()[_name] = getattr(_m, _cls)
        __all__.append(_name)
    except Exception as _e:
        print(f'[modelhub] 模型 {_name} 导入失败，已跳过（不影响其他模型）: {_e}')
