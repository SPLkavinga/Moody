# live_emotion_inference.py
import cv2
import math
import time
import joblib
import argparse
import numpy as np
from collections import deque, Counter
from typing import Dict, List, Tuple, Optional

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh


# MUST match training feature order (without 'emotion')

FEATURE_ORDER = [
    'mouth_movement', 'mouth_aspect_ratio', 'lip_corner_distance', 'jaw_drop',
    'left_eye_movement', 'right_eye_movement', 'left_eyebrow_movement', 'right_eyebrow_movement',
    'left_eyebrow_slope', 'right_eyebrow_slope', 'eyebrow_asymmetry', 'nostril_flare',
    'nose_tip_movement', 'left_cheek_position', 'right_cheek_position', 'jaw_width',
    'mouth_eye_ratio', 'pose_yaw', 'pose_pitch', 'pose_roll', 'left_eye_ear',
    'right_eye_ear', 'eye_asymmetry', 'interocular_norm', 'mouth_corner_slope',
    'mouth_curvature', 'smile_intensity', 'brow_eye_dist_left',
    'brow_eye_dist_right', 'brow_eye_asymmetry', 'cheek_asymmetry', 'jaw_angle_deg',
    'nose_to_mouth', 'nose_to_chin_norm', 'face_width_norm', 'face_height_norm', 'face_wh_ratio'
]


# Math, geometry helpers

def safe_L(landmarks: List[Tuple[int,int,float]], i: int) -> Optional[Tuple[int,int,float]]:
    return None if (i < 0 or i >= len(landmarks)) else landmarks[i]

def sdist(a: Optional[Tuple[int,int,float]], b: Optional[Tuple[int,int,float]]) -> Optional[float]:
    if a is None or b is None:
        return None
    return math.hypot(b[0] - a[0], b[1] - a[1])

def sratio(num: Optional[float], den: Optional[float]) -> float:
    if num is None or den in (None, 0):
        return 0.0
    return float(num) / float(den)

def angle_deg(a: Optional[Tuple[int,int,float]], 
              b: Optional[Tuple[int,int,float]], 
              c: Optional[Tuple[int,int,float]]) -> float:
    if a is None or b is None or c is None:
        return 0.0
    v1 = (a[0]-b[0], a[1]-b[1])
    v2 = (c[0]-b[0], c[1]-b[1])
    n1 = math.hypot(*v1); n2 = math.hypot(*v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = max(-1.0, min(1.0, (v1[0]*v2[0] + v1[1]*v2[1]) / (n1*n2)))
    return math.degrees(math.acos(cosang))

def point_line_signed_distance(p: Optional[Tuple[int,int,float]],
                               a: Optional[Tuple[int,int,float]],
                               b: Optional[Tuple[int,int,float]]) -> float:
    if p is None or a is None or b is None:
        return 0.0
    x0,y0 = p[0], p[1]
    x1,y1 = a[0], a[1]
    x2,y2 = b[0], b[1]
    A = y1 - y2
    B = x2 - x1
    C = x1*y2 - x2*y1
    denom = math.hypot(A, B)
    if denom == 0:
        return 0.0
    return (A*x0 + B*y0 + C) / denom


# Head pose estimation via solvePnP

def estimate_head_pose(landmarks: List[Tuple[int,int,float]], w: int, h: int) -> Tuple[float,float,float]:
    idxs = [1, 152, 33, 263, 61, 291]
    pts2d = []
    for i in idxs:
        p = safe_L(landmarks, i)
        if p is None:
            return 0.0, 0.0, 0.0
        pts2d.append([float(p[0]), float(p[1])])
    pts2d = np.array(pts2d, dtype=np.float64)

    pts3d = np.array([
        [  0.0,   0.0,   0.0],   # nose tip
        [  0.0, -90.0, -10.0],   # chin
        [-60.0,  40.0, -30.0],   # left eye outer
        [ 60.0,  40.0, -30.0],   # right eye outer
        [-40.0, -40.0, -30.0],   # left mouth
        [ 40.0, -40.0, -30.0],   # right mouth
    ], dtype=np.float64)

    focal = w
    center = (w / 2.0, h / 2.0)
    cam_mtx = np.array([[focal, 0, center[0]],
                        [0, focal, center[1]],
                        [0,    0,        1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(pts3d, pts2d, cam_mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0,0]*R[0,0] + R[1,0]*R[1,0])
    singular = sy < 1e-6

    if not singular:
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        yaw   = math.degrees(math.atan2(R[1,0], R[0,0]))
        roll  = math.degrees(math.atan2(R[2,1], R[2,2]))
    else:
        pitch = math.degrees(math.atan2(-R[2,0], sy))
        yaw   = math.degrees(math.atan2(-R[0,1], R[1,1]))
        roll  = 0.0

    return yaw, pitch, roll


# Compute features for a single face

def compute_features(landmarks: List[Tuple[int,int,float]], w: int, h: int) -> Dict[str, float]:
    L = lambda i: safe_L(landmarks, i)

    total_reference = sdist(L(4), L(6))
    mouth_width_ref = sdist(L(61), L(291))
    left_eye_width_ref  = sdist(L(33), L(133))
    right_eye_width_ref = sdist(L(362), L(263))

    # Mouth
    mouth_height = sdist(L(13), L(14))
    mouth_aspect_ratio = sratio(mouth_height, mouth_width_ref)
    lip_corner_distance = sratio(mouth_width_ref, total_reference)
    jaw_drop = sratio(sdist(L(152), L(14)), total_reference)

    # Eyes
    left_eye_height  = sdist(L(159), L(145))
    right_eye_height = sdist(L(386), L(374))
    left_eye_openness  = sratio(left_eye_height, left_eye_width_ref) / (total_reference if total_reference else 1)
    right_eye_openness = sratio(right_eye_height, right_eye_width_ref) / (total_reference if total_reference else 1)

    # Brows
    left_eyebrow_height  = sratio(sdist(L(65),  L(33)),  left_eye_width_ref)  / (total_reference if total_reference else 1)
    right_eyebrow_height = sratio(sdist(L(295), L(263)), right_eye_width_ref) / (total_reference if total_reference else 1)
    left_eyebrow_slope  = ((L(159)[1] - L(65)[1])  / (L(159)[0] - L(65)[0]))  if (L(159) and L(65)  and (L(159)[0]-L(65)[0]))   else 0.0
    right_eyebrow_slope = ((L(386)[1] - L(295)[1]) / (L(386)[0] - L(295)[0])) if (L(386) and L(295) and (L(386)[0]-L(295)[0])) else 0.0
    eyebrow_asymmetry = abs(left_eyebrow_height - right_eyebrow_height)

    # Nose
    nostril_flare = sratio(sdist(L(98), L(327)), total_reference)
    nose_tip_movement = sratio(sdist(L(1), L(4)), total_reference)

    # Cheeks / Jaw
    left_cheek_pos  = sratio(sdist(L(230), L(295)), total_reference)
    right_cheek_pos = sratio(sdist(L(450), L(426)), total_reference)
    jaw_width = sratio(sdist(L(234), L(454)), total_reference)

    # Derived
    eye_distance = sratio(sdist(L(33), L(263)), total_reference)
    mouth_eye_ratio = sratio(mouth_height, (eye_distance if eye_distance else None))

    #  Head pose
    yaw, pitch, roll = estimate_head_pose(landmarks, w, h)

    # eye EAR  asymmetry
    left_eye_ear  = sratio(left_eye_height, left_eye_width_ref)
    right_eye_ear = sratio(right_eye_height, right_eye_width_ref)
    eye_asymmetry = abs(left_eye_ear - right_eye_ear)
    interocular_norm = sratio(sdist(L(33), L(263)), total_reference)

    # mouth slope/curvature/smile
    mouth_corner_slope = 0.0
    if L(61) and L(291) and (L(291)[0] - L(61)[0]) != 0:
        mouth_corner_slope = (L(291)[1] - L(61)[1]) / (L(291)[0] - L(61)[0])
    mouth_mid = ((L(13)[0]+L(14)[0])//2, (L(13)[1]+L(14)[1])//2, 0.0) if (L(13) and L(14)) else None
    mouth_curvature = abs(point_line_signed_distance(mouth_mid, L(61), L(291))) if mouth_mid else 0.0
    mouth_curvature = sratio(mouth_curvature, mouth_width_ref)
    smile_signed = point_line_signed_distance(mouth_mid, L(61), L(291)) if mouth_mid else 0.0
    smile_intensity = sratio(smile_signed, mouth_width_ref)
    # up_d = abs(point_line_signed_distance(L(13), L(61), L(291))) if L(13) else 0.0
    # lo_d = abs(point_line_signed_distance(L(14), L(61), L(291))) if L(14) else 0.0
    # upper_lower_lip_ratio = sratio(up_d, (lo_d if lo_d else None))

    # brow eye distances
    left_eye_center  = None if (L(33) is None or L(133) is None) else ((L(33)[0]+L(133)[0])//2, (L(33)[1]+L(133)[1])//2, 0.0)
    right_eye_center = None if (L(362) is None or L(263) is None) else ((L(362)[0]+L(263)[0])//2, (L(362)[1]+L(263)[1])//2, 0.0)
    brow_eye_dist_left  = sratio(sdist(L(65),  left_eye_center),  left_eye_width_ref)  / (total_reference if total_reference else 1)
    brow_eye_dist_right = sratio(sdist(L(295), right_eye_center), right_eye_width_ref) / (total_reference if total_reference else 1)
    brow_eye_asymmetry = abs(brow_eye_dist_left - brow_eye_dist_right)

    # cheek asymmetry and jaw angle
    cheek_asymmetry = abs(left_cheek_pos - right_cheek_pos)
    jaw_angle_deg = angle_deg(L(234), L(152), L(454))

    # nose distances
    nose_to_mouth = sratio(sdist(L(1), L(13)), total_reference)
    nose_to_chin_norm = sratio(sdist(L(1), L(152)), total_reference)

    # global shape
    face_width = sdist(L(234), L(454))
    face_height = sdist(L(10), L(152)) if (L(10) and L(152)) else sdist(L(1), L(152))
    face_width_norm = sratio(face_width, total_reference)
    face_height_norm = sratio(face_height, total_reference)
    face_wh_ratio = sratio(face_width, (face_height if face_height else None))

    feat = {
        'mouth_movement': sratio(mouth_height, total_reference),
        'mouth_aspect_ratio': mouth_aspect_ratio,
        'lip_corner_distance': lip_corner_distance,
        'jaw_drop': jaw_drop,
        'left_eye_movement': left_eye_openness,
        'right_eye_movement': right_eye_openness,
        'left_eyebrow_movement': left_eyebrow_height,
        'right_eyebrow_movement': right_eyebrow_height,
        'left_eyebrow_slope': left_eyebrow_slope,
        'right_eyebrow_slope': right_eyebrow_slope,
        'eyebrow_asymmetry': eyebrow_asymmetry,
        'nostril_flare': nostril_flare,
        'nose_tip_movement': nose_tip_movement,
        'left_cheek_position': left_cheek_pos,
        'right_cheek_position': right_cheek_pos,
        'jaw_width': jaw_width,
        'mouth_eye_ratio': mouth_eye_ratio,
        'pose_yaw': yaw,
        'pose_pitch': pitch,
        'pose_roll': roll,
        'left_eye_ear': left_eye_ear,
        'right_eye_ear': right_eye_ear,
        'eye_asymmetry': eye_asymmetry,
        'interocular_norm': interocular_norm,
        'mouth_corner_slope': mouth_corner_slope,
        'mouth_curvature': mouth_curvature,
        'smile_intensity': smile_intensity,
        # 'upper_lower_lip_ratio' upper lower lip ratio,
        'brow_eye_dist_left': brow_eye_dist_left,
        'brow_eye_dist_right': brow_eye_dist_right,
        'brow_eye_asymmetry': brow_eye_asymmetry,
        'cheek_asymmetry': cheek_asymmetry,
        'jaw_angle_deg': jaw_angle_deg,
        'nose_to_mouth': nose_to_mouth,
        'nose_to_chin_norm': nose_to_chin_norm,
        'face_width_norm': face_width_norm,
        'face_height_norm': face_height_norm,
        'face_wh_ratio': face_wh_ratio
    }
    return feat

# Build feature vector in correct order

def vectorize_features(feat_dict: Dict[str, float]) -> np.ndarray:
    return np.array([feat_dict.get(name, 0.0) for name in FEATURE_ORDER], dtype=np.float32).reshape(1, -1)


# Live webcam loop

def run_live(model_path: str, labels_path: str, cam_index: int = 0,
             window: int = 10, min_det_conf: float = 0.5, refine: bool = False):
    # Load model pipeline and label encoder
    model = joblib.load(model_path)
    le = joblib.load(labels_path)

    # Prediction smoother
    recent = deque(maxlen=window)

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {cam_index}")
        return

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=refine,
        min_detection_confidence=min_det_conf,
        min_tracking_confidence=0.5
    ) as face_mesh:

        prev_t = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            pred_label = None
            conf = None  # for models that provide predict_proba
            if result.multi_face_landmarks:
                # Take the first face
                fl = result.multi_face_landmarks[0]
                # Collect 468 landmarks
                landmarks = []
                for lm in fl.landmark:
                    x_px = int(round(lm.x * w))
                    y_px = int(round(lm.y * h))
                    z_px = lm.z * w
                    landmarks.append((x_px, y_px, z_px))

                # Compute features and predict
                feats = compute_features(landmarks, w, h)
                X = vectorize_features(feats)

                # Predict
                try:
                    y_pred = model.predict(X)[0]
                    pred_label = le.inverse_transform([y_pred])[0]
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X)[0]
                        conf = float(np.max(proba))
                except Exception as e:
                    pred_label = f"ERR: {e}"

            # Temporal smoothing
            if pred_label is not None and not str(pred_label).startswith("ERR"):
                recent.append(pred_label)
                most_common, count = Counter(recent).most_common(1)[0]
                smoothed_label = most_common
            else:
                smoothed_label = None

            # FPS
            now = time.time()
            fps = 1.0 / (now - prev_t) if now > prev_t else 0.0
            prev_t = now

            # Draw overlays
            overlay = frame.copy()
            y0 = 30
            cv2.putText(overlay, f"FPS: {fps:.1f}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            y0 += 25

            if smoothed_label:
                txt = f"Emotion: {smoothed_label}"
                if conf is not None:
                    txt += f" ({conf:.2f})"
                cv2.putText(overlay, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                y0 += 25
            elif pred_label is not None:
                cv2.putText(overlay, f"Prediction: {pred_label}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                y0 += 25
            else:
                cv2.putText(overlay, "No face detected", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            cv2.imshow("Live Emotion Detection", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or q to quit
                break

    cap.release()
    cv2.destroyAllWindows()


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live webcam emotion detection using a saved model pipeline.")
    parser.add_argument("--model", required=True, help="Path to saved emotion model pipeline (emotion_model.joblib).")
    parser.add_argument("--labels", required=True, help="Path to saved LabelEncoder (label_encoder.joblib).")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default: 0).")
    parser.add_argument("--smooth", type=int, default=10, help="Temporal smoothing window size in frames (default: 10).")
    parser.add_argument("--min_det_conf", type=float, default=0.5, help="MediaPipe min_detection_confidence (default: 0.5).")
    parser.add_argument("--refine", action="store_true", help="Use refine_landmarks=True (slower, slightly better iris/eye).")
    args = parser.parse_args()

    run_live(args.model, args.labels, cam_index=args.cam, window=args.smooth,
             min_det_conf=args.min_det_conf, refine=args.refine)
