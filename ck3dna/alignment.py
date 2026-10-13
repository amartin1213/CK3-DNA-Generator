"""ck3dna.alignment

Standard dlib 68-landmark face alignment: detect the largest face in an
image, estimate a similarity transform from its eye landmarks, and warp the
image to a fixed-size square crop with eyes placed at a canonical position.

This is a from-scratch re-implementation of the common "align_face" recipe
used throughout face-processing pipelines (the upstream repo's
`utils/alignment.py` was not fully reviewed by the user requesting this
rebuild, so nothing was copied from it -- this is an independent
derivation of the same standard technique).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import dlib
import numpy as np

# Canonical landmark indices for the 68-point iBUG face model.
_LEFT_EYE = list(range(36, 42))
_RIGHT_EYE = list(range(42, 48))

_face_detector = None
_shape_predictors: dict = {}


def _get_detector():
    global _face_detector
    if _face_detector is None:
        _face_detector = dlib.get_frontal_face_detector()
    return _face_detector


def _get_predictor(predictor_path: str):
    predictor = _shape_predictors.get(predictor_path)
    if predictor is None:
        predictor = dlib.shape_predictor(predictor_path)
        _shape_predictors[predictor_path] = predictor
    return predictor


def _shape_to_np(shape) -> np.ndarray:
    return np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)


def align_face(
    filepath,
    predictor_path: str,
    output_size: int = 256,
    left_eye_pos=(0.35, 0.35),
) -> Optional[np.ndarray]:
    """Detect, align and crop the largest face found in `filepath`.

    Args:
        filepath: path to the source image.
        predictor_path: path to dlib's `shape_predictor_68_face_landmarks.dat`
            (see README.md for how to obtain this file; it is not shipped
            in this repo).
        output_size: side length, in pixels, of the square output crop.
        left_eye_pos: desired (x, y) position of the left eye within the
            output crop, as a fraction of `output_size`. The right eye is
            placed at the mirrored x position, same y.

    Returns:
        A BGR `output_size` x `output_size` numpy array, or `None` if no
        face was detected.
    """
    image = cv2.imread(str(filepath))
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    detector = _get_detector()
    predictor = _get_predictor(predictor_path)

    detections = detector(gray, 1)
    if len(detections) == 0:
        return None
    # Use the largest detected face (by bounding-box area).
    rect = max(detections, key=lambda r: r.width() * r.height())

    landmarks = _shape_to_np(predictor(gray, rect))
    left_eye_center = landmarks[_LEFT_EYE].mean(axis=0)
    right_eye_center = landmarks[_RIGHT_EYE].mean(axis=0)

    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    angle = np.degrees(np.arctan2(dy, dx))

    right_eye_pos_x = 1.0 - left_eye_pos[0]
    current_dist = np.hypot(dx, dy)
    desired_dist = (right_eye_pos_x - left_eye_pos[0]) * output_size
    scale = desired_dist / current_dist if current_dist > 1e-6 else 1.0

    eyes_center = (
        float((left_eye_center[0] + right_eye_center[0]) / 2.0),
        float((left_eye_center[1] + right_eye_center[1]) / 2.0),
    )

    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)
    target_x = output_size * 0.5
    target_y = output_size * left_eye_pos[1]
    rotation_matrix[0, 2] += target_x - eyes_center[0]
    rotation_matrix[1, 2] += target_y - eyes_center[1]

    aligned = cv2.warpAffine(
        image, rotation_matrix, (output_size, output_size), flags=cv2.INTER_CUBIC
    )
    return aligned


def align_and_save(filepath, predictor_path: str, output_path, output_size: int = 256) -> bool:
    """Align `filepath` and write the result to `output_path`.

    Returns True on success, False if no face was detected.
    """
    aligned = align_face(filepath, predictor_path, output_size=output_size)
    if aligned is None:
        return False
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), aligned)
    return True
