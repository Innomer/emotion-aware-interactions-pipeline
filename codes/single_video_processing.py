from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch

mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

def extract_focused_frames(
    video_path: str,
    num_frames: int = 3,
    padding_ratio: float = 0.15,
    visualize: bool = False,
) -> List[Image.Image]:

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise ValueError(f"No frames found in video: {video_path}")

    speaker_indices = (
        np.linspace(
            total_frames * 0.1,
            total_frames * 0.9,
            min(24, total_frames),
        )
        .astype(int)
        .tolist()
    )

    speaker_frames = []

    for frame_idx in speaker_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_idx,
        )

        success, frame = cap.read()

        if success:
            speaker_frames.append((frame_idx, frame))

    speaker_track, face_visualizations = _find_speaker(
        speaker_frames,
        visualize,
    )

    if num_frames == 1:

        clip_indices = [total_frames // 2]

    else:

        clip_indices = (
            np.linspace(
                total_frames * 0.1,
                total_frames * 0.9,
                min(num_frames, total_frames),
            )
            .astype(int)
            .tolist()
        )

    pose_detector = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        min_detection_confidence=0.5,
    )

    frames = []
    pose_visualizations = []

    for frame_idx in clip_indices:

        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(frame_idx),
        )

        success, frame_bgr = cap.read()

        if not success:
            continue

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        face_box = None

        if speaker_track is not None:

            face_box = _get_speaker_box(
                speaker_track,
                frame_idx,
            )

        if face_box is None:

            final_crop = frame_rgb

            if visualize:

                pose_visualizations.append(
                    (
                        frame_rgb.copy(),
                        frame_rgb.copy(),
                        final_crop.copy(),
                    )
                )

        else:

            person_region_box = _expand_face_box(
                face_box,
                frame_rgb.shape,
            )

            pose_box, pose_landmarks = _detect_person(
                frame_rgb,
                pose_detector,
                face_box,
            )

            if pose_box is not None:

                final_box = _merge_boxes(
                    person_region_box,
                    pose_box,
                )

                final_box = _add_padding(
                    final_box,
                    frame_rgb.shape,
                    padding_ratio,
                )

            else:

                final_box = _add_padding(
                    person_region_box,
                    frame_rgb.shape,
                    padding_ratio,
                )

            final_crop = _crop_box(
                frame_rgb,
                final_box,
            )

            if visualize:

                face_vis = frame_rgb.copy()

                _draw_box(
                    face_vis,
                    face_box,
                    (0, 255, 0),
                    "SPEAKER FACE",
                    2,
                )

                _draw_box(
                    face_vis,
                    person_region_box,
                    (255, 165, 0),
                    "PERSON REGION",
                    2,
                )

                pose_vis = frame_rgb.copy()

                if pose_landmarks is not None:

                    mp_drawing.draw_landmarks(
                        pose_vis,
                        pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=(
                            mp_drawing_styles.get_default_pose_landmarks_style()
                        ),
                    )

                if pose_box is not None:

                    _draw_box(
                        pose_vis,
                        pose_box,
                        (255, 0, 0),
                        "POSE",
                        2,
                    )

                _draw_box(
                    pose_vis,
                    final_box,
                    (0, 255, 0),
                    "FINAL",
                    3,
                )

                pose_visualizations.append(
                    (
                        face_vis,
                        pose_vis,
                        final_crop.copy(),
                    )
                )

        frames.append(Image.fromarray(final_crop))

    cap.release()
    pose_detector.close()

    if visualize:

        _show_face_visualization(face_visualizations)

        _show_pose_visualization(pose_visualizations)

    if not frames:

        raise ValueError(f"Could not extract frames from: {video_path}")

    return frames


def _find_speaker(
    frames: List[Tuple[int, np.ndarray]],
    visualize: bool,
):

    if not frames:
        return None, []

    face_detector = mp_face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.1,
    )

    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.1,
        min_tracking_confidence=0.1,
    )

    tracks: Dict[int, Dict] = {}

    next_track_id = 0

    visualizations = []

    for actual_frame_idx, frame_bgr in frames:

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB,
        )

        h, w = frame_rgb.shape[:2]

        detection_result = face_detector.process(frame_rgb)

        current_faces = []

        if detection_result.detections:

            for detection in detection_result.detections:

                bbox = detection.location_data.relative_bounding_box

                x1 = int(bbox.xmin * w)

                y1 = int(bbox.ymin * h)

                x2 = int((bbox.xmin + bbox.width) * w)

                y2 = int((bbox.ymin + bbox.height) * h)

                x1 = max(
                    0,
                    min(w - 1, x1),
                )

                y1 = max(
                    0,
                    min(h - 1, y1),
                )

                x2 = max(
                    0,
                    min(w, x2),
                )

                y2 = max(
                    0,
                    min(h, y2),
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                box = (
                    x1,
                    y1,
                    x2,
                    y2,
                )

                center = np.array(
                    [
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                    ]
                )

                current_faces.append(
                    {
                        "box": box,
                        "center": center,
                        "confidence": detection.score[0],
                        "frame_idx": actual_frame_idx,
                    }
                )

        for face_data in current_faces:

            x1, y1, x2, y2 = face_data["box"]

            face_crop = frame_rgb[
                y1:y2,
                x1:x2,
            ]

            if face_crop.size == 0:
                continue

            mesh_result = face_mesh.process(face_crop)

            if not mesh_result.multi_face_landmarks:
                continue

            # FaceMesh should find the face inside this crop.
            landmarks = mesh_result.multi_face_landmarks[0]

            mouth_left = landmarks.landmark[61]
            mouth_right = landmarks.landmark[291]
            mouth_top = landmarks.landmark[13]
            mouth_bottom = landmarks.landmark[14]

            mouth_width = np.linalg.norm(
                np.array(
                    [
                        mouth_left.x,
                        mouth_left.y,
                    ]
                )
                - np.array(
                    [
                        mouth_right.x,
                        mouth_right.y,
                    ]
                )
            )

            mouth_height = np.linalg.norm(
                np.array(
                    [
                        mouth_top.x,
                        mouth_top.y,
                    ]
                )
                - np.array(
                    [
                        mouth_bottom.x,
                        mouth_bottom.y,
                    ]
                )
            )

            mouth_ratio = mouth_height / (mouth_width + 1e-6)

            face_data["mouth"] = mouth_ratio
            face_data["landmarks"] = landmarks

        # Remove detections where FaceMesh failed
        current_faces = [face for face in current_faces if "mouth" in face]

        for face in current_faces:

            best_track = None
            best_distance = float("inf")

            for track_id, track in tracks.items():

                if not track["detections"]:
                    continue

                previous = track["detections"][-1]

                previous_box = previous["box"]

                previous_width = previous_box[2] - previous_box[0]

                distance = np.linalg.norm(face["center"] - previous["center"])

                if distance < previous_width * 2.0 and distance < best_distance:

                    best_distance = distance
                    best_track = track_id

            if best_track is None:

                best_track = next_track_id
                next_track_id += 1

                tracks[best_track] = {
                    "detections": [],
                    "motion": [],
                }

            track = tracks[best_track]

            if track["detections"]:

                previous_mouth = track["detections"][-1]["mouth"]

                motion = abs(face["mouth"] - previous_mouth)

                track["motion"].append(motion)

            track["detections"].append(face)

        # -----------------------------------------------------
        # VISUALIZATION
        # -----------------------------------------------------

        if visualize:

            vis = frame_rgb.copy()

            for face in current_faces:

                _draw_box(
                    vis,
                    face["box"],
                    (0, 255, 0),
                    f"Face {face['confidence']:.2f}",
                    2,
                )

                # Draw landmarks if available
                if "landmarks" in face:

                    # Convert cropped-face landmarks back to
                    # full-frame coordinates.

                    crop_landmarks = face["landmarks"]

                    face_x1, face_y1, _, _ = face["box"]

                    for landmark in crop_landmarks.landmark:

                        lx = int(
                            face_x1 + landmark.x * (face["box"][2] - face["box"][0])
                        )

                        ly = int(
                            face_y1 + landmark.y * (face["box"][3] - face["box"][1])
                        )

                        cv2.circle(
                            vis,
                            (lx, ly),
                            1,
                            (255, 0, 0),
                            -1,
                        )

            visualizations.append(
                (
                    actual_frame_idx,
                    vis,
                )
            )

    face_detector.close()
    face_mesh.close()

    valid_tracks = {
        track_id: track
        for track_id, track in tracks.items()
        if len(track["detections"]) >= 3
    }

    if not valid_tracks:

        return None, visualizations

    speaker_id = max(
        valid_tracks,
        key=lambda track_id: _speaker_score(valid_tracks[track_id]),
    )

    speaker_track = valid_tracks[speaker_id]

    # ---------------------------------------------------------
    # MARK SPEAKER IN VISUALIZATION
    # ---------------------------------------------------------

    if visualize:

        updated_visualizations = []

        for frame_idx, vis in visualizations:

            for detection in speaker_track["detections"]:

                if detection["frame_idx"] == frame_idx:

                    _draw_box(
                        vis,
                        detection["box"],
                        (255, 0, 0),
                        "SPEAKER",
                        3,
                    )

                    break

            updated_visualizations.append(
                (
                    frame_idx,
                    vis,
                )
            )

        visualizations = updated_visualizations

    return (
        speaker_track,
        visualizations,
    )


def _speaker_score(track) -> float:

    if not track["motion"]:
        return 0.0

    motion = np.asarray(
        track["motion"],
        dtype=np.float32,
    )

    return float(np.mean(motion) + 0.5 * np.std(motion))


def _get_speaker_box(
    speaker_track,
    frame_idx: int,
):

    detections = speaker_track["detections"]

    if not detections:
        return None

    exact = [
        detection for detection in detections if detection["frame_idx"] == frame_idx
    ]

    if exact:
        return exact[0]["box"]

    closest = min(
        detections,
        key=lambda detection: abs(detection["frame_idx"] - frame_idx),
    )

    return closest["box"]


def _expand_face_box(
    box,
    shape,
):

    h, w = shape[:2]

    x1, y1, x2, y2 = box

    face_width = x2 - x1
    face_height = y2 - y1

    return (
        max(
            0,
            int(x1 - face_width * 0.8),
        ),
        max(
            0,
            int(y1 - face_height * 0.5),
        ),
        min(
            w,
            int(x2 + face_width * 0.8),
        ),
        min(
            h,
            int(y2 + face_height * 3.5),
        ),
    )


def _detect_person(
    frame_rgb,
    detector,
    face_box,
):

    result = detector.process(frame_rgb)

    if not result.pose_landmarks:
        return None, None

    h, w = frame_rgb.shape[:2]

    landmarks = [lm for lm in result.pose_landmarks.landmark if lm.visibility > 0.5]

    if not landmarks:
        return None, None

    xs = [lm.x * w for lm in landmarks]

    ys = [lm.y * h for lm in landmarks]

    pose_box = (
        max(0, int(min(xs))),
        max(0, int(min(ys))),
        min(w, int(max(xs))),
        min(h, int(max(ys))),
    )

    fx1, fy1, fx2, fy2 = face_box

    face_center = np.array(
        [
            (fx1 + fx2) / 2,
            (fy1 + fy2) / 2,
        ]
    )

    px1, py1, px2, py2 = pose_box

    pose_center = np.array(
        [
            (px1 + px2) / 2,
            (py1 + py2) / 2,
        ]
    )

    pose_width = px2 - px1
    pose_height = py2 - py1

    distance = np.linalg.norm(face_center - pose_center)

    if (
        distance
        > max(
            pose_width,
            pose_height,
        )
        * 0.75
    ):

        return (
            None,
            result.pose_landmarks,
        )

    return (
        pose_box,
        result.pose_landmarks,
    )


def _merge_boxes(
    box1,
    box2,
):

    return (
        min(box1[0], box2[0]),
        min(box1[1], box2[1]),
        max(box1[2], box2[2]),
        max(box1[3], box2[3]),
    )


def _add_padding(
    box,
    shape,
    padding_ratio,
):

    h, w = shape[:2]

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio

    return (
        max(
            0,
            int(x1 - pad_x),
        ),
        max(
            0,
            int(y1 - pad_y),
        ),
        min(
            w,
            int(x2 + pad_x),
        ),
        min(
            h,
            int(y2 + pad_y),
        ),
    )


def _crop_box(
    frame_rgb,
    box,
):

    x1, y1, x2, y2 = box

    return frame_rgb[
        y1:y2,
        x1:x2,
    ]


def _draw_box(
    image,
    box,
    color,
    label,
    thickness=2,
):

    x1, y1, x2, y2 = box

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness,
    )

    cv2.putText(
        image,
        label,
        (x1, max(25, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def _show_face_visualization(
    visualizations,
):

    if not visualizations:
        return

    count = len(visualizations)

    cols = min(4, count)

    rows = int(np.ceil(count / cols))

    plt.figure(figsize=(16, 4 * rows))

    for i, (
        frame_idx,
        frame,
    ) in enumerate(visualizations):

        plt.subplot(
            rows,
            cols,
            i + 1,
        )

        plt.imshow(frame)

        plt.axis("off")

        plt.title(f"Face Detection + Landmarks\n" f"Frame {frame_idx}")

    plt.tight_layout()
    plt.show()


def _show_pose_visualization(
    visualizations,
):

    if not visualizations:
        return

    count = len(visualizations)

    plt.figure(figsize=(15, 4 * count))

    for i, (
        face_vis,
        pose_vis,
        final_crop,
    ) in enumerate(visualizations):

        plt.subplot(
            count,
            3,
            i * 3 + 1,
        )

        plt.imshow(face_vis)

        plt.axis("off")

        plt.title("Speaker Face")

        plt.subplot(
            count,
            3,
            i * 3 + 2,
        )

        plt.imshow(pose_vis)

        plt.axis("off")

        plt.title("MediaPipe Pose")

        plt.subplot(
            count,
            3,
            i * 3 + 3,
        )

        plt.imshow(final_crop)

        plt.axis("off")

        plt.title("Final CLIP Crop")

    plt.tight_layout()
    plt.show()


def process_video(
    video_path=r"D:\emotion-aware-interactions-pipeline\data\MELD.Raw\train_splits\dia0_utt0.mp4",
    visualize=False,
    clip_frames=3,
):

    frames = extract_focused_frames(
        video_path,
        num_frames=clip_frames,
        visualize=visualize,
    )

    clip_inputs = processor(
        images=frames,
        return_tensors="pt",
    )

    print(
        f"Extracted {len(frames)} frames, "
        f"pixel_values shape: "
        f"{clip_inputs['pixel_values'].shape}"
    )

    try:
        with torch.no_grad():
            outputs = clip_model.vision_model(pixel_values=clip_inputs["pixel_values"])
            # image_embeddings = clip_model.visual_projection(outputs.pooler_output)
            patch_embeddings = outputs.last_hidden_state
            patch_embeddings = outputs.last_hidden_state[:, 1:, :]  # removing CLS

            print("Patch embedding shape:", patch_embeddings.shape)

            B, N, D = patch_embeddings.shape

            patch_grid = patch_embeddings.reshape(
                B,
                7,
                7,
                D,
            )

            # 4 spatial regions
            top_left = patch_grid[:, :3, :3, :]
            top_right = patch_grid[:, :3, 3:, :]
            bottom_left = patch_grid[:, 3:, :3, :]
            bottom_right = patch_grid[:, 3:, 3:, :]

            # Spatial average pooling
            token_1 = top_left.mean(dim=(1, 2))
            token_2 = top_right.mean(dim=(1, 2))
            token_3 = bottom_left.mean(dim=(1, 2))
            token_4 = bottom_right.mean(dim=(1, 2))

            visual_tokens = torch.stack(
                [
                    token_1,
                    token_2,
                    token_3,
                    token_4,
                ],
                dim=1,
            )

            print("Image Tokened: ", visual_tokens.shape)
            return visual_tokens
    except Exception as e:
        print("Error in CLIP: ", e)
        return None
