import io
import os
import tempfile
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import requests
import streamlit as st

# Optional: real-time camera capture
try:
    from streamlit_webrtc import WebRtcMode, webrtc_streamer
except Exception:  # pragma: no cover - optional
    webrtc_streamer = None
    WebRtcMode = None

# API config
API_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
VN_TZ = timezone(timedelta(hours=7))


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #
def _vn_now() -> datetime:
    return datetime.now(VN_TZ)


def _fmt_dt(value: Optional[str | datetime]) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except Exception:
            return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=VN_TZ)
    return value.astimezone(VN_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _post_multipart(endpoint: str, file_bytes: bytes, filename: str, fields: Dict[str, Any]) -> requests.Response:
    files = {"file": (filename, io.BytesIO(file_bytes))}
    return requests.post(f"{API_URL}{endpoint}", files=files, data=fields, timeout=180)


def _render_results(results: List[Dict[str, Any]]):
    if not results:
        st.info("Không phát hiện khuôn mặt.")
        return
    for item in results:
        st.write(
            f"**Name**: {item.get('name', 'unknown')} | "
            f"Confidence: {item.get('confidence')} | "
            f"Appeared at: {_fmt_dt(item.get('appeared_at'))}"
        )


def _image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _build_video_from_frames(frames: List[np.ndarray], fps: int = 15) -> tuple[bytes, str]:
    """
    Build a short MJPG/AVI clip from captured frames to avoid ffmpeg codec issues.
    """
    if not frames:
        raise ValueError("No frames captured from camera")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp:
        video_path = tmp.name

    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        os.remove(video_path)
        raise RuntimeError("Cannot open VideoWriter (codec MJPG).")

    for frame in frames:
        writer.write(frame)
    writer.release()

    with open(video_path, "rb") as f:
        data = f.read()
    os.remove(video_path)
    return data, os.path.basename(video_path)


def _build_video_from_image(image_bytes: bytes, fps: int = 10, duration_sec: float = 1.0) -> tuple[bytes, str]:
    """
    Dùng 1 frame từ camera để dựng 1 video ngắn (lặp lại frame) gửi lên API detect/enroll video.
    """
    frame = _image_bytes_to_bgr(image_bytes)
    if frame is None:
        raise ValueError("Cannot decode camera image")

    h, w = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    total_frames = max(1, int(fps * duration_sec))

    with tempfile.NamedTemporaryFile(delete=False, suffix=".avi") as tmp:
        video_path = tmp.name

    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        os.remove(video_path)
        raise RuntimeError("Cannot open VideoWriter (codec MJPG).")

    for _ in range(total_frames):
        writer.write(frame)
    writer.release()

    with open(video_path, "rb") as f:
        data = f.read()
    os.remove(video_path)
    return data, os.path.basename(video_path)


def _fetch_attendance(kind: str, target_date: date) -> Dict[str, Any]:
    endpoint = {
        "on_time_arrivals": "/attendance/on-time-arrivals",
        "late_arrivals": "/attendance/late-arrivals",
        "early_leaves": "/attendance/early-leaves",
        "on_time_leaves": "/attendance/on-time-leaves",
    }[kind]
    resp = requests.get(f"{API_URL}{endpoint}", params={"target_date": target_date.isoformat()}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _add_manual_appearance(name: str, appeared_at: datetime) -> Dict[str, Any]:
    payload = {
        "name": name,
        "appeared_at": appeared_at.isoformat(),
    }
    resp = requests.post(f"{API_URL}/attendance/appearances", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_stats() -> Dict[str, Any]:
    resp = requests.get(f"{API_URL}/attendance/stats", timeout=30)
    resp.raise_for_status()
    return resp.json()


# --------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------- #
st.set_page_config(page_title="Face Recognition", page_icon="🧠", layout="wide")
st.title("Face Recognition Demo (YOLO + ArcFace)")
st.caption(f"API: {API_URL}")

tab_detect_image, tab_detect_video, tab_attendance, tab_enroll = st.tabs(
    ["Detect Image", "Detect Video", "Attendance", "Enroll Person"]
)

# Detect image
with tab_detect_image:
    st.subheader("Upload Image")
    source_name = st.text_input("Source name (optional)", value="")
    captured = st.camera_input("Chụp từ camera", key="cam_image")
    if captured and st.button("Run detection", type="primary"):
        with st.spinner("Sending to API..."):
            resp = _post_multipart(
                "/images/detect",
                captured.getvalue(),
                captured.name or "camera.jpg",
                {"source_name": source_name},
            )
        if resp.ok:
            data = resp.json()
            st.success("Detection completed")
            _render_results(data.get("results", []))
        else:
            st.error(f"API error: {resp.status_code} - {resp.text}")

# Detect video
with tab_detect_video:
    st.subheader("Upload Video")
    captured_video_bytes: Optional[bytes] = None

    if webrtc_streamer:
        st.caption("Quay trực tiếp từ camera (dùng streamlit-webrtc)")
        cam_state = st.session_state.setdefault("detect_video_frames", deque(maxlen=180))

        def _video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            cam_state.append(img)
            return frame

        webrtc_streamer(
            key="detect_video_webrtc",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=_video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )

        try:
            if len(cam_state) >= 30:  # đủ ~2s ở 15fps
                captured_video_bytes, video_name = _build_video_from_frames(list(cam_state))
        except Exception as exc:
            st.error(f"Không tạo được video từ camera: {exc}")
    else:
        st.warning("Thiếu streamlit-webrtc, fallback dùng 1 frame camera lặp lại.")
        cam_video_frame = st.camera_input("Quay nhanh từ camera (dùng 1 frame lặp lại)", key="cam_video")
        if cam_video_frame:
            try:
                captured_video_bytes, video_name = _build_video_from_image(cam_video_frame.getvalue())
            except Exception as exc:
                st.error(f"Không tạo được video từ camera: {exc}")

    if captured_video_bytes and st.button("Run video detection", type="primary", key="run_video"):
        with st.spinner("Processing video via API..."):
            resp = _post_multipart(
                "/videos/detect",
                captured_video_bytes,
                video_name,
                {},
            )
        if resp.ok:
            data = resp.json()
            st.success("Video processed")
            _render_results(data.get("results", []))
        else:
            st.error(f"API error: {resp.status_code} - {resp.text}")

# Attendance tab
with tab_attendance:
    st.subheader("Attendance lookup (giờ Việt Nam)")
    target_day = st.date_input("Chọn ngày", value=_vn_now().date())
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Đến đúng giờ (<8h)", key="btn_on_time_arrive"):
            try:
                data = _fetch_attendance("on_time_arrivals", target_day)
                st.success(f"Tìm thấy {len(data.get('people', []))} người đến đúng giờ")
                _render_results(data.get("people", []))
            except Exception as exc:
                st.error(f"Lỗi: {exc}")
        if st.button("Đi muộn (8h-12h)", key="btn_late_arrive"):
            try:
                data = _fetch_attendance("late_arrivals", target_day)
                st.success(f"Tìm thấy {len(data.get('people', []))} người đi muộn")
                _render_results(data.get("people", []))
            except Exception as exc:
                st.error(f"Lỗi: {exc}")
    with col_b:
        if st.button("Về sớm (12h-<17h)", key="btn_early_leave"):
            try:
                data = _fetch_attendance("early_leaves", target_day)
                st.success(f"Tìm thấy {len(data.get('people', []))} người về sớm")
                _render_results(data.get("people", []))
            except Exception as exc:
                st.error(f"Lỗi: {exc}")
        if st.button("Về đúng giờ (>=17h)", key="btn_on_time_leave"):
            try:
                data = _fetch_attendance("on_time_leaves", target_day)
                st.success(f"Tìm thấy {len(data.get('people', []))} người về đúng giờ")
                _render_results(data.get("people", []))
            except Exception as exc:
                st.error(f"Lỗi: {exc}")

    st.markdown("---")
    st.subheader("Thêm thủ công lần xuất hiện")
    col1, col2 = st.columns(2)
    with col1:
        manual_name = st.text_input("Tên người", key="manual_name")
    with col2:
        manual_time = st.time_input("Thời gian (VN)", value=_vn_now().time(), key="manual_time")
    manual_date = st.date_input("Ngày (VN)", value=_vn_now().date(), key="manual_date")
    if st.button("Lưu appearance", type="primary", key="btn_manual_add"):
        if not manual_name:
            st.error("Nhập tên trước khi lưu.")
        else:
            appeared_at = datetime.combine(manual_date, manual_time, tzinfo=VN_TZ)
            try:
                data = _add_manual_appearance(manual_name, appeared_at)
                st.success(f"Đã lưu appearance cho {data.get('name')}")
                st.write(f"Arrived at: {_fmt_dt(data.get('arrived_at'))}")
            except Exception as exc:
                st.error(f"Lỗi: {exc}")

    st.markdown("---")
    st.subheader("Thống kê on-time / late / early-leave")
    if st.button("Lấy thống kê", key="btn_stats"):
        try:
            data = _fetch_stats()
            stats = data.get("stats", [])
            if not stats:
                st.info("Chưa có dữ liệu thống kê.")
            else:
                st.table(
                    [
                        {
                            "Name": item.get("name"),
                            "On time days": item.get("on_time_days"),
                            "Late days": item.get("late_days"),
                            "Early-leave days": item.get("early_leave_days"),
                            "Updated": _fmt_dt(item.get("updated_at")),
                        }
                        for item in stats
                    ]
                )
        except Exception as exc:
            st.error(f"Lỗi: {exc}")

# Enroll tab
with tab_enroll:
    st.subheader("Enroll person from video")
    person_name = st.text_input("Person name", key="enroll_name")
    captured_enroll_video: Optional[bytes] = None

    if webrtc_streamer:
        st.caption("Quay trực tiếp từ camera để enroll (streamlit-webrtc)")
        enroll_state = st.session_state.setdefault("enroll_video_frames", deque(maxlen=180))

        def _enroll_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            enroll_state.append(img)
            return frame

        webrtc_streamer(
            key="enroll_video_webrtc",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=_enroll_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=False,
        )
        try:
            if len(enroll_state) >= 30:  # đủ ~2s ở 15fps
                captured_enroll_video, enroll_video_name = _build_video_from_frames(list(enroll_state))
        except Exception as exc:
            st.error(f"Không tạo được video từ camera: {exc}")
    else:
        st.warning("Thiếu streamlit-webrtc, fallback dùng 1 frame camera lặp lại.")
        cam_enroll_frame = st.camera_input("Quay nhanh từ camera để enroll", key="enroll_video_cam")
        if cam_enroll_frame:
            try:
                captured_enroll_video, enroll_video_name = _build_video_from_image(cam_enroll_frame.getvalue())
            except Exception as exc:
                st.error(f"Không tạo được video từ camera: {exc}")

    if captured_enroll_video and person_name and st.button("Enroll from video", type="primary", key="run_enroll_video"):
        with st.spinner("Uploading to API..."):
            resp = _post_multipart(
                "/persons/enroll-video",
                captured_enroll_video,
                enroll_video_name,
                {"name": person_name},
            )
        if resp.ok:
            data = resp.json()
            st.success(f"Enroll thành công: id={data.get('id')}, name={data.get('name')}, samples={data.get('samples')}")
        else:
            st.error(f"API error: {resp.status_code} - {resp.text}")
    elif captured_enroll_video and not person_name:
        st.info("Nhập tên trước khi enroll.")
