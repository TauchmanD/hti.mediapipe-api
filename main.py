import os
import tempfile

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

MODEL_PATH = "./pose.task"  # make sure this file is in the container/workdir

app = FastAPI()


def draw_landmarks_on_image(rgb_image, detection_result):
    annotated = np.copy(rgb_image)
    for pose_landmarks in detection_result.pose_landmarks:
        proto = landmark_pb2.NormalizedLandmarkList()
        proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=l.x,
                y=l.y,
                z=l.z,
                visibility=getattr(l, "visibility", 0.0),
            )
            for l in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated,
            proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated


def get_pose_landmarker():
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def open_video(input_path: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or np.isnan(fps):
        fps = 30.0  # safe default

    if width == 0 or height == 0:
        raise RuntimeError("Input reports 0x0 dimensions. Re-encode your input first.")

    return cap, width, height, fps


def process_video(input_path: str, output_path: str):
    cap, width, height, fps = open_video(input_path)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError("Could not open VideoWriter. Check codecs / ffmpeg.")

    MPImage = mp.Image
    with get_pose_landmarker() as landmarker:
        frame_index = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_index / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            annotated_rgb = draw_landmarks_on_image(frame_rgb, result)
            annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)

            out.write(annotated_bgr)
            frame_index += 1

    cap.release()
    out.release()


def extract_points_from_video(input_path: str):
    """
    Returns a list of frames with pose points:
    [
      {
        "frame_index": int,
        "timestamp_ms": int,
        "poses": [
          [
            {"x": float, "y": float, "z": float, "visibility": float},
            ...
          ]
        ]
      },
      ...
    ]
    """
    cap, width, height, fps = open_video(input_path)

    MPImage = mp.Image
    frames_data = []

    with get_pose_landmarker() as landmarker:
        frame_index = 0
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = MPImage(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((frame_index / fps) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            poses = []
            for pose_landmarks in result.pose_landmarks:
                points = [
                    {
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": getattr(lm, "visibility", 0.0),
                    }
                    for lm in pose_landmarks
                ]
                poses.append(points)

            frames_data.append(
                {
                    "frame_index": frame_index,
                    "timestamp_ms": timestamp_ms,
                    "poses": poses,
                }
            )
            frame_index += 1

    cap.release()
    return frames_data


@app.post("/annotate")
async def annotate_video(file: UploadFile = File(...)):
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".mp4"  # fallback

    # Save uploaded file to temp
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
            in_path = tmp_in.name
            content = await file.read()
            tmp_in.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input video: {e}")

    # Prepare output temp file
    out_fd, out_path = tempfile.mkstemp(suffix=".avi")
    os.close(out_fd)

    try:
        process_video(in_path, out_path)
    except Exception as e:
        # Clean up on error
        os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)
        raise HTTPException(status_code=500, detail=f"Video processing failed: {e}")

    # Remove input; remove output after response
    os.remove(in_path)

    def cleanup(path: str):
        if os.path.exists(path):
            os.remove(path)

    return FileResponse(
        out_path,
        media_type="video/x-msvideo",
        filename=f"annotated_{os.path.basename(file.filename).rsplit('.', 1)[0]}.avi",
        background=BackgroundTask(cleanup, out_path),
    )


@app.post("/points")
async def get_points(file: UploadFile = File(...)):
    """
    Returns pose points for each frame as JSON.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    ext = os.path.splitext(file.filename)[1]
    if not ext:
        ext = ".mp4"  # fallback

    # Save uploaded file to temp
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_in:
            in_path = tmp_in.name
            content = await file.read()
            tmp_in.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save input video: {e}")

    try:
        frames_data = extract_points_from_video(in_path)
    except Exception as e:
        os.remove(in_path)
        raise HTTPException(status_code=500, detail=f"Point extraction failed: {e}")

    os.remove(in_path)
    return {"frames": frames_data}

