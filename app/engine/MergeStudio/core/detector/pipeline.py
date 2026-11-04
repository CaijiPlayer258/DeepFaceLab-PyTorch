"""
Complete face detection + landmark + alignment pipeline.
Reference: Extractor/Extractor.py
"""
import sys
from pathlib import Path
import cv2, math
import numpy as np
import facelib
from facelib.LandmarksProcessor import get_transform_mat
from MergeStudio.core.detector.landmarks import landmark106to68

# Ensure project root is in sys.path for modelhub
_proj_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if _proj_root not in sys.path:
    sys.path.insert(0, _proj_root)


def _is_self_padding_detector(detector) -> bool:
    """YoloV8Face and RetinaFace handle their own image padding."""
    try:
        from modelhub.onnx.YoloV8Face import YoloV8Face
        from modelhub.onnx.RetinaFace import RetinaFace
        return isinstance(detector, (YoloV8Face, RetinaFace))
    except Exception:
        return False


# ── FaceMesh 468 canonical landmarks（用于 umeyama 仿射对齐）──
_FACEMESH_CANONICAL = np.float32([
  [0.4999769926, 0.6525340080], [0.5000259876, 0.5474870205], [0.4999740124, 0.6023719907],
  [0.4821130037, 0.4719790220], [0.5001509786, 0.5271559954], [0.4999099970, 0.4982529879],
  [0.4995230138, 0.4010620117], [0.2897120118, 0.3807640076], [0.4999549985, 0.3123980165],
  [0.4999870062, 0.2699189782], [0.5000230074, 0.1070500016], [0.5000230074, 0.6662340164],
  [0.5000159740, 0.6792240143], [0.5000230074, 0.6923480034], [0.4999769926, 0.6952779889],
  [0.4999769926, 0.7059339881], [0.4999769926, 0.7193850279], [0.4999769926, 0.7370190024],
  [0.4999679923, 0.7813709974], [0.4998160005, 0.5629810095], [0.4737730026, 0.5739099979],
  [0.1049069986, 0.2541409731], [0.3659299910, 0.4095759988], [0.3387579918, 0.4130250216],
  [0.3111200035, 0.4094600081], [0.2746579945, 0.3891310096], [0.3933619857, 0.4037060142],
  [0.3452340066, 0.3440110087], [0.3700940013, 0.3460760117], [0.3193219900, 0.3472650051],
  [0.2979030013, 0.3535910249], [0.2477920055, 0.4108099937], [0.3968890011, 0.8427550197],
  [0.2800979912, 0.3755999804], [0.1063100025, 0.3999559879], [0.2099249959, 0.3913530111],
  [0.3558079898, 0.5344060063], [0.4717510045, 0.6504039764], [0.4741550088, 0.6801919937],
  [0.4397850037, 0.6572290063], [0.4146170020, 0.6665409803], [0.4503740072, 0.6808609962],
  [0.4287709892, 0.6826909781], [0.3749710023, 0.7278050184], [0.4867169857, 0.5476289988],
  [0.4853009880, 0.5273950100], [0.2577649951, 0.3144900203], [0.4012230039, 0.4551720023],
  [0.4298189878, 0.5486149788], [0.4213519990, 0.5337409973], [0.2768959999, 0.5320569873],
  [0.4833700061, 0.4995869994], [0.3372119963, 0.2828829885], [0.2963919938, 0.2932429910],
  [0.1692949980, 0.1938139796], [0.4475800097, 0.3026099801], [0.3923900127, 0.3538879752],
  [0.3544900119, 0.6967840195], [0.0673049986, 0.7301050425], [0.4427390099, 0.5728260279],
  [0.4570980072, 0.5847920179], [0.3819740117, 0.6947109699], [0.3923889995, 0.6942030191],
  [0.2770760059, 0.2719320059], [0.4225519896, 0.5632330179], [0.3859190047, 0.2813640237],
  [0.3831030130, 0.2558400035], [0.3314310014, 0.1197140217], [0.2299239933, 0.2320029736],
  [0.3645009995, 0.1891139746], [0.2296220064, 0.2995409966], [0.1732870042, 0.2787479758],
  [0.4728789926, 0.6661980152], [0.4468280077, 0.6685270071], [0.4227620065, 0.6738899946],
  [0.4453079998, 0.5800659657], [0.3881030083, 0.6939610243], [0.4030390084, 0.7065399885],
  [0.4036290050, 0.6939530373], [0.4600419998, 0.5571390390], [0.4311580062, 0.6923660040],
  [0.4521819949, 0.6923660040], [0.4753870070, 0.6923660040], [0.4658280015, 0.7791900039],
  [0.4723289907, 0.7362259626], [0.4730870128, 0.7178570032], [0.4731220007, 0.7046259642],
  [0.4730330110, 0.6952779889], [0.4279420078, 0.6952779889], [0.4264790118, 0.7035399675],
  [0.4231620133, 0.7118459940], [0.4183090031, 0.7200629711], [0.3900949955, 0.6395729780],
  [0.0139539996, 0.5600340366], [0.4999139905, 0.5801470280], [0.4131999910, 0.6953999996],
  [0.4096260071, 0.7018229961], [0.4680800140, 0.6015349627], [0.4227289855, 0.5859850049],
  [0.4630799890, 0.5937839746], [0.3721199930, 0.4734140038], [0.3345620036, 0.4960730076],
  [0.4116710126, 0.5469650030], [0.2421759963, 0.1476759911], [0.2907769978, 0.2014459968],
  [0.3273380101, 0.2565270066], [0.3995099962, 0.7489210367], [0.4417279959, 0.2616760135],
  [0.4297649860, 0.1878340244], [0.4121980071, 0.1089010239], [0.2889550030, 0.3989520073],
  [0.2189369947, 0.4354109764], [0.4127820134, 0.3989700079], [0.2571350038, 0.3554400206],
  [0.4276849926, 0.4379609823], [0.4483399987, 0.5369360447], [0.1785600036, 0.4575539827],
  [0.2473080009, 0.4571939707], [0.2862670124, 0.4676749706], [0.3328279853, 0.4607120156],
  [0.3687559962, 0.4472069740], [0.3989639878, 0.4326549768], [0.4764100015, 0.4058060050],
  [0.1892410070, 0.5239239931], [0.2289620042, 0.3489509821], [0.4907259941, 0.5624009967],
  [0.4046700001, 0.4851329923], [0.0194690004, 0.4015640020], [0.4262430072, 0.4204310179],
  [0.3969930112, 0.5487970114], [0.2664699852, 0.3769770265], [0.4391210079, 0.5189579725],
  [0.0323139988, 0.6443569660], [0.4190540016, 0.3871549964], [0.4627830088, 0.5057469606],
  [0.2389789969, 0.7797449827], [0.1982209980, 0.8319380283], [0.1075500026, 0.5407550335],
  [0.1836100072, 0.7402570248], [0.1344099939, 0.3336830139], [0.3857640028, 0.8831539750],
  [0.4909670055, 0.5793780088], [0.3823849857, 0.5085729957], [0.1743990034, 0.3976709843],
  [0.3187850118, 0.3962349892], [0.3433640003, 0.4005969763], [0.3961000144, 0.7102169991],
  [0.1878850013, 0.5885379910], [0.4309870005, 0.9440649748], [0.3189930022, 0.8982850313],
  [0.2662479877, 0.8697010279], [0.5000230074, 0.1905760169], [0.4999769926, 0.9544529915],
  [0.3661699891, 0.3988220096], [0.3932070136, 0.3955370188], [0.4103730023, 0.3910800219],
  [0.1949930042, 0.3421019912], [0.3886649907, 0.3622840047], [0.3659619987, 0.3559709787],
  [0.3433640003, 0.3553569913], [0.3187850118, 0.3583400249], [0.3014149964, 0.3631560206],
  [0.0581329986, 0.3190760016], [0.3014149964, 0.3874490261], [0.4999879897, 0.6184340119],
  [0.4158380032, 0.6241959929], [0.4456819892, 0.5660769939], [0.4658440053, 0.6206409931],
  [0.4999229908, 0.3515239954], [0.2887189984, 0.8199459910], [0.3352789879, 0.8528199792],
  [0.4405120015, 0.9024189711], [0.1282940060, 0.7919409871], [0.4087719917, 0.3738939762],
  [0.4556069970, 0.4518010020], [0.4998770058, 0.9089900255], [0.3754369915, 0.9241920114],
  [0.1142100021, 0.6150220037], [0.4486620128, 0.6952779889], [0.4480200112, 0.7046320438],
  [0.4471119940, 0.7158080339], [0.4448319972, 0.7307940125], [0.4300119877, 0.7668089867],
  [0.4067870080, 0.6856729984], [0.4007380009, 0.6810690165], [0.3923999965, 0.6777030230],
  [0.3678559959, 0.6639189720], [0.2479230016, 0.6013330221], [0.4527699947, 0.4208499789],
  [0.4363920093, 0.3598870039], [0.4161640108, 0.3687139750], [0.4133859873, 0.6923660040],
  [0.2280180007, 0.6835719943], [0.4682680070, 0.3526710272], [0.4113619924, 0.8043270111],
  [0.4999890029, 0.4698250294], [0.4791539907, 0.4426540136], [0.4999740124, 0.4396370053],
  [0.4321120083, 0.4935889840], [0.4998860061, 0.8669170141], [0.4999130070, 0.8217290044],
  [0.4565489888, 0.8192009926], [0.3445490003, 0.7454389930], [0.3789089918, 0.5740100145],
  [0.3742929995, 0.7801849842], [0.3196879923, 0.5707379580], [0.3571549952, 0.6042699814],
  [0.2952840030, 0.6215809584], [0.4477500021, 0.8624770045], [0.4109860063, 0.5087230206],
  [0.3139509857, 0.7753080130], [0.3541280031, 0.8125529885], [0.3245480061, 0.7039929628],
  [0.1890960038, 0.6462999582], [0.2797769904, 0.7146580219], [0.1338230073, 0.6827009916],
  [0.3367680013, 0.6447330117], [0.4298839867, 0.4665219784], [0.4555279911, 0.5486229658],
  [0.4371140003, 0.5588960052], [0.4672879875, 0.5299249887], [0.4147120118, 0.3352199793],
])

def facemesh_to_align_mat(landmarks_468: np.ndarray, image_size: tuple) -> np.ndarray:
    """FaceMesh 468 点 → affine transform (umeyama)."""
    h, w = image_size[:2]
    dst = _FACEMESH_CANONICAL * (w, h)
    src = landmarks_468[:, :2].astype(np.float64) if landmarks_468.shape[1] >= 2 else landmarks_468.astype(np.float64)
    if src.shape[0] > 400:
        src = src[:400]
        dst = dst[:400]
    from core.mathlib.umeyama import umeyama
    T = umeyama(src, dst, True)
    return T[:2]


def _detect_faces_multi_angle(detector, image: np.ndarray, angles: list) -> list:
    """Detect faces after rotating image by each angle, merge unique detections."""
    h, w = image.shape[:2]
    all_detections = []
    for angle in angles:
        if angle == 0:
            rotated_img = image
        elif angle == 90:
            rotated_img = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated_img = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            continue

        raw = detector.extract(rotated_img)
        if not raw:
            continue
        faces = raw[0] if isinstance(raw[0], list) else raw
        for face_rect in faces:
            l, t, r, b = face_rect[:4]
            if angle == 90:
                ol, ot, or_, ob = t, w - r, b, w - l
            elif angle == 180:
                ol, ot, or_, ob = w - r, h - b, w - l, h - t
            elif angle == 270:
                ol, ot, or_, ob = h - b, l, h - t, r
            else:
                ol, ot, or_, ob = l, t, r, b
            ol = max(0, int(ol)); ot = max(0, int(ot))
            or_ = min(w, int(or_)); ob = min(h, int(ob))
            if or_ > ol and ob > ot:
                all_detections.append((angle, ol, ot, or_, ob))
    # Deduplicate by IoU
    if not all_detections:
        return []
    all_detections.sort(key=lambda x: (x[3]-x[1])*(x[4]-x[2]), reverse=True)
    keep = []
    for d in all_detections:
        _, l1, t1, r1, b1 = d
        dup = False
        for k in keep:
            _, l2, t2, r2, b2 = k
            il = max(l1, l2); it = max(t1, t2)
            ir = min(r1, r2); ib = min(b1, b2)
            if ir > il and ib > it:
                inter = (ir-il)*(ib-it)
                union = (r1-l1)*(b1-t1) + (r2-l2)*(b2-t2) - inter
                if union > 0 and inter/union > 0.5:
                    dup = True; break
        if not dup:
            keep.append(d)
    return keep


def detect_and_align(detector, landmarker, image: np.ndarray,
                     face_type_str: str = 'whole_face',
                     fixed_window: int = 0,
                     margin: float = 0.4,
                     detection_angles: list = None):
    h_orig, w_orig = image.shape[:2]
    scale_factor = 1.0
    working_image = image

    if fixed_window > 0 and w_orig > fixed_window:
        scale_factor = w_orig / fixed_window
        new_h = int(h_orig / scale_factor)
        working_image = cv2.resize(image, (fixed_window, new_h),
                                   interpolation=cv2.INTER_AREA)

    # Determine detection angles (default: single 0-degree pass)
    _angles = detection_angles if detection_angles else [0]
    _has_multi = len(_angles) > 1 or _angles[0] != 0

    if _has_multi:
        # Multi-angle detection: detect on rotated versions, merge results
        # Returns [(angle, l, t, r, b), ...]
        _rotated_faces = _detect_faces_multi_angle(detector, working_image, _angles)
        face_list = [(box, None, angle) for (angle, *box) in _rotated_faces]
    else:
        # Single angle (0°) — existing optimized path with KPS support
        if hasattr(detector, 'extract_with_kps'):
            raw = detector.extract_with_kps(working_image)
            face_list = raw
        else:
            raw = detector.extract(working_image)
            raw = raw[0] if isinstance(raw[0], list) else raw
            face_list = [(box, None) for box in raw]

    if not face_list or len(face_list) == 0:
        return []

    face_data_list = []
    for item in face_list:
        # Unpack: multi-angle items have 3 elements (box, kps, det_angle)
        if len(item) == 3:
            face_rect, kps, det_angle = item
        else:
            face_rect, kps = item if isinstance(item, tuple) else (item, None)
            det_angle = 0
        l, t, r, b = face_rect[:4]
        l, t, r, b = int(l), int(t), int(r), int(b)

        # Square crop centered on face (based on larger dimension + margin)
        fw, fh = r - l, b - t
        base = max(fw, fh)
        half = int(base * (0.5 + margin))
        cx, cy = (l + r) // 2, (t + b) // 2
        l_crop = max(0, cx - half)
        t_crop = max(0, cy - half)
        r_crop = min(working_image.shape[1], cx + half)
        b_crop = min(working_image.shape[0], cy + half)

        face_img = working_image[t_crop:b_crop, l_crop:r_crop]
        if face_img.size == 0:
            continue

        # Compute effective pre-rotation angle (KPS keypoints OR detection angle)
        rot_angle = 0.0
        if kps is not None and len(kps) >= 5:
            eye_cx = (kps[0][0] + kps[1][0]) / 2.0
            eye_cy = (kps[0][1] + kps[1][1]) / 2.0
            mc_x = (kps[3][0] + kps[4][0]) / 2.0
            mc_y = (kps[3][1] + kps[4][1]) / 2.0
            rot_angle = math.degrees(math.atan2(eye_cx - mc_x, -(eye_cy - mc_y)))
        elif det_angle != 0:
            # Face detected at non-zero angle — rotate crop back upright
            rot_angle = det_angle
        if abs(rot_angle) > 30.0:
            h_f, w_f = face_img.shape[:2]
            center = (w_f // 2, h_f // 2)
            rot_mat = cv2.getRotationMatrix2D(center, rot_angle, 1.0)
            face_img = cv2.warpAffine(face_img, rot_mat, (w_f, h_f), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

        lmks = None
        try:
            landmark_results = landmarker.extract(face_img)
            if landmark_results is not None and len(landmark_results) > 0:
                pts = landmark_results[0].copy()
                if abs(rot_angle) > 30.0:
                    h_f2, w_f2 = face_img.shape[:2]
                    center2 = (w_f2 // 2, h_f2 // 2)
                    rot_inv = cv2.getRotationMatrix2D(center2, -rot_angle, 1.0)
                    ones = np.ones((pts.shape[0], 1))
                    pts = (rot_inv @ np.concatenate([pts, ones], axis=1).T).T
                pts[:, 0] += l_crop
                pts[:, 1] += t_crop
                lmks = pts
        except Exception:
            continue

        if lmks is None:
            continue

        is_facemesh = len(lmks) == 468

        if len(lmks) == 106:
            lmks = landmark106to68(lmks)
        elif len(lmks) > 68 and not is_facemesh:
            # InsightFace3D68 (3309 pts): first 68 are standard landmarks
            lmks = lmks[:68]

        lmks_orig = lmks.copy()
        lmks_orig[:, 0] *= scale_factor
        lmks_orig[:, 1] *= scale_factor

        face_type_map = {
            'half_face': facelib.FaceType.HALF,
            'midfull_face': facelib.FaceType.MID_FULL,
            'full_face': facelib.FaceType.FULL,
            'whole_face': facelib.FaceType.WHOLE_FACE,
            'head': facelib.FaceType.HEAD,
        }
        face_type_enum = face_type_map.get(face_type_str, facelib.FaceType.WHOLE_FACE)
        out_size = 256

        if is_facemesh:
            # FaceMesh 468: use Umeyama with canonical landmarks
            mat = facemesh_to_align_mat(lmks, (h_orig, w_orig))
        else:
            mat = get_transform_mat(lmks_orig, out_size, face_type_enum)

        face_data_list.append({
            'face_rect': (int(l * scale_factor), int(t * scale_factor),
                          int(r * scale_factor), int(b * scale_factor)),
            'crop_rect': (int(l_crop * scale_factor), int(t_crop * scale_factor),
                          int(r_crop * scale_factor), int(b_crop * scale_factor)),
            'landmarks': lmks_orig,
            'transform_mat': mat,
            'out_size': out_size,
            'face_type': face_type_str,
            'face_type_enum': face_type_enum,
        })

    return face_data_list


def apply_alignment(image: np.ndarray, face_data: dict) -> np.ndarray:
    mat = face_data.get('transform_mat')
    if mat is None:
        mat = get_transform_mat(face_data['landmarks'], face_data['out_size'],
                                face_data['face_type_enum'])
    return cv2.warpAffine(image, mat, (face_data['out_size'], face_data['out_size']),
                          flags=cv2.INTER_LANCZOS4)
