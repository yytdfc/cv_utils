import cv2
import torch
import numpy as np
import PIL
from PIL import Image
from PIL import ImageDraw
import face_alignment

from .cast import cast


device = "cuda" if torch.cuda.is_available() else "cpu"


__MAX_LOAD_DECTOR = 2


def get_face_detector(name="blazeface_back", _face_detectors={}):
    if name not in _face_detectors:
        if len(_face_detectors) >= __MAX_LOAD_DECTOR:
            _face_detectors.pop(next(iter(_face_detectors)))
        if name == "sfd":
            _face_detectors[name] = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False,
                device=device, face_detector='sfd')
        elif name == "blazeface_back":
            _face_detectors[name] = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False,
                                            device=device, face_detector='blazeface', face_detector_kwargs={'back_model': True})
        elif name == "blazeface":
            _face_detectors[name] = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False,
                                            device=device, face_detector='blazeface')
        elif name == "dlib":
            _face_detectors[name] = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D, flip_input=False,
                                            device=device, face_detector='dlib')
        else:
            raise ValueError(f"face detector {name} not support")
        
        print("Initialize", name)

    return _face_detectors[name]


def perspective_transform(img, points, source_coords, target_coords, target_size):
    # original
    # (x0, y0)  \ /  (x1, y1)
    #            x
    # (x3, y3)  / \  (x2, y2)
    # perspective_transform
    # (x0, y0)  \ /  (x1, y1)
    #            x
    # (x3, y3)  / \  (x2, y2)
    #  matrix = []
    #  for s, t in zip(source_coords, target_coords):
        #  matrix.append([t[0], t[1], 1, 0, 0, 0, -s[0] * t[0], -s[0] * t[1]])
        #  matrix.append([0, 0, 0, t[0], t[1], 1, -s[1] * t[0], -s[1] * t[1]])
    #  A = np.matrix(matrix, dtype=np.float32)
    #  B = np.array(source_coords).reshape(8)
    #  coeffs = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    #  coeffs = np.array(coeffs).reshape(8)
    #  m = np.append(coeffs, 1).reshape((3, 3)).copy()
    #  img = img.transform(target_size, Image.PERSPECTIVE, coeffs, Image.BILINEAR)

    src = np.array(source_coords, dtype=np.float32)
    dest = np.array(target_coords, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dest)
    img = cv2.warpPerspective(img, matrix, target_size, cv2.INTER_AREA)

    if len(points):
        p = np.array([[px, py, 1] for px, py in points], dtype=np.float32)
        p = matrix.dot(p.T)
        p = p[:2] / p[2]
        points = p.T
    return img, points


def ffhq_align(img, lm, output_size, scale=1.0, rotate=False):
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    if rotate:
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.0 * scale, np.hypot(*eye_to_mouth) * 1.8 * scale)
        y = np.flipud(x) * [-1, 1]
        c0 = eye_avg + eye_to_mouth * 0.1
    else:
        x = np.array([1, 0], dtype=np.float64)
        x *= max(np.hypot(*eye_to_eye) * 2.0 * scale, np.hypot(*eye_to_mouth) * 1.8 * scale)
        y = np.flipud(x) * [-1, 1]
        c0 = eye_avg + eye_to_mouth * 0.1

    source_coords = np.stack([c0 - x - y, c0 + x - y, c0 + x + y, c0 - x + y])
    if isinstance(output_size, int):
        output_w, output_h = output_size, output_size
    else:
        output_w, output_h = output_size
    target_coords = [(0, 0), (output_w - 1, 0), (output_w - 1, output_h - 1), (0, output_h - 1)]
    img, lm = perspective_transform(img, lm, source_coords, target_coords, (output_w, output_h))
    return img, lm, [source_coords, target_coords]


def align_back(im_align, size, transform_paras):
    im_align_back, _ = perspective_transform(im_align, [], transform_paras[1], transform_paras[0], size)
    # make mask with landmarks
    w, h = size
    #  mouth_polygon = np.concatenate([preds[0][3:13], preds[0][35:31:-1]],0)
    #  print(mouth_polygon)
    mask = np.zeros((h, w), np.uint8)
    #  mask_0 = cv2.fillPoly(mask, np.int32([transform_paras[0]]), 255, cv2.INTER_LINEAR)
    mask = cv2.fillPoly(mask, np.int32([transform_paras[0]]), 255)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    return im_align_back, mask


def detect(im, detector="blazeface_back"):
    #  im = cast(im, "np")
    fa = get_face_detector(detector)
    if isinstance(im, list):
        im = torch.cat([cast(i, "tensor") for i in im], 0)
        lm = fa.get_landmarks_from_batch(im)
    elif im.ndim == 4:
        im = cast(im, "tensor")
        lm = fa.get_landmarks_from_batch(im)
    else:
        lm = fa.get_landmarks(im)
    return lm


def detect_and_align(im, output_size, rotate=False, detector="blazeface_back", prev_box=None, scale=1.0):
    im = cast(im, "np")

    if prev_box is not None:
        x0, y0, x1, y1 = prev_box
        if im.ndim == 4:
            im0 = im[:, y0:y1, x0:x1, :]
        else:
            im0 = im[y0:y1, x0:x1, :]
    else:
        im0 = im

    lm = detect(im0, detector=detector)

    if lm is None:
        return None

    if prev_box is not None:
        for l in lm:
            if len(l):
                l += np.array([x0, y0])

    if isinstance(im, list) or im.ndim == 4:
        return [ffhq_align(i, l, output_size=output_size, rotate=rotate, scale=scale) if l is not None else None for i, l in zip(im, lm)]
    else:
        return ffhq_align(im, lm[0], output_size=output_size, rotate=rotate, scale=scale)


def smooth_bbox(bboxs, window=5):
    # bbox [
    #  [(x0, y0), (x1, y1), (x2, y2), (x3, y3)],
    #  [(x0, y0), (x1, y1), (x2, y2), (x3, y3)],
    # ]

    bboxs = np.array(bboxs)
    # center_x, center_y, width, height
    bboxs = np.array([
        (bboxs[:, 0, 0] + bboxs[:, 2, 0]) / 2, 
        (bboxs[:, 0, 1] + bboxs[:, 2, 1]) / 2, 
        bboxs[:, 2, 0] - bboxs[:, 0, 0],
        bboxs[:, 2, 1] - bboxs[:, 0, 1],
    ])
    half = window // 2
    bboxs = np.pad(bboxs, ((0, 0), (half, window - half)), mode="edge")
    bboxs = np.cumsum(bboxs, axis=1)
    bboxs = (bboxs[:, window:] - bboxs[:, :-window]) / window
    # bboxs = bboxs[:, half:half-window] / window

    return bboxs.T


def get_mask_from_lm(lms, region="mouth", size=(512, 512)):
    h, w = size
    mask = np.zeros((h, w), np.uint8)
    if region == "mouth":
        polygon = np.concatenate([lms[3:13], lms[35:31:-1]], 0)
    elif region == "nose":
        polygon = lms[27:36]
    elif region == "eye":
        polygon = np.concatenate([lms[36:42], lms[42:48]], 0)
    elif region == "eyebrow":
        polygon = np.concatenate([lms[17:22], lms[22:27]], 0)
    else:
        raise ValueError(f"region {region} not support")
    mask = cv2.fillPoly(mask, np.int32([polygon]), 255)
    return mask


if __name__ == "__main__":
    pass
    fa = get_face_detector("blazeface_back")
    lm = fa.get_landmarks(np.array(Image.open("/Users/cfu/git/101_lipsync/SadTalker/examples/1 copy.png")))
    print(lm)

