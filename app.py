# app.py — Streamlit + WebRTC + RTSP + Image Upload + Demo Video with YOLOv8 + HuggingFace Auto Download
# ---------------------------------------------------------------------------------------------------------
# 功能：
# - 4 種來源：手機後鏡頭(WebRTC)、RTSP/IP Cam、單張圖片、示範影片(伺服器側循環)
# - 模型來源支援：Hugging Face 自動下載（預設啟用）、本地/上傳 .pt、安全回退 yolov8n.pt
# - 類別過濾、conf/IoU 調整、FPS 與偵測數量疊字、模型/類別資訊面板
# ---------------------------------------------------------------------------------------------------------

import os
import time
from typing import Optional, List

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# ------------------------------
# Streamlit Page Config (一定要在任何 st.* 之前)
# ------------------------------
st.set_page_config(page_title="YOLO Homecage Mice — HF SafeLoad", page_icon="🐭", layout="wide")
st.set_option("client.showErrorDetails", False)  # 隱藏粉紅 Network issue 區塊

HF_TOKEN = os.getenv("HF_TOKEN_FOR_STREAMLIT", None)

# ------------------------------
# Sidebar Controls
# ------------------------------
st.sidebar.title("⚙️ 偵測設定")

# 模型來源選擇（預設 Hugging Face）
st.sidebar.subheader("🧠 模型來源")
model_source = st.sidebar.radio(
    "選擇模型來源",
    options=["本地/上傳 .pt", "Hugging Face Model Hub"],
    index=1  # 預設選 Hugging Face
)

# 通用閾值設定
conf_thres = st.sidebar.slider("置信度閾值 (conf)", 0.05, 0.9, 0.35, 0.05)
iou_thres = st.sidebar.slider("IoU 閾值 (NMS)", 0.1, 0.9, 0.5, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 類別過濾（選填）")
classes_text = st.sidebar.text_input("只保留類別（以逗號分隔）", value="", help="例如：0,1,2；留空=不過濾")

# 視訊來源模式
st.sidebar.markdown("---")
st.sidebar.subheader("📹 視訊來源")
mode = st.sidebar.radio(
    "選擇來源",
    options=["WebRTC（手機後鏡頭）", "RTSP / IP Cam", "單張圖片上傳測試", "示範影片（伺服器側）"],
    index=0
)

# 本地/上傳 .pt 欄位
model_path_input, uploaded_model = None, None
# Hugging Face 欄位（預設為你的 repo 與檔名）
hf_repo_id, hf_filename, hf_revision = None, None, None

if model_source == "本地/上傳 .pt":
    st.sidebar.markdown("**本地/上傳 .pt**")
    default_model_hint = "（例如：best.pt 或 /app/best.pt；留空會回退 yolov8n.pt）"
    model_path_input = st.sidebar.text_input("模型路徑 / 檔名", value="", help=default_model_hint)
    uploaded_model = st.sidebar.file_uploader("或上傳本地 .pt 模型檔", type=["pt"], help="上傳後會暫存為 ./uploaded_model.pt 並優先使用")
else:
    st.sidebar.markdown("**Hugging Face Model Hub**")
    hf_repo_id = st.sidebar.text_input("Repo ID", value="Mei-1206/YOLO_homecage_mice", help="例如：Mei-1206/YOLO_homecage_mice")
    hf_filename = st.sidebar.text_input("檔名（repo 內）", value="best.pt", help="例如：best.pt 或 weights/best.pt")
    hf_revision = st.sidebar.text_input("Revision（選填）", value="", help="tag / branch / commit；留空用預設")
    if not hf_repo_id.strip():
        st.error("請填入有效的 Hugging Face Repo ID。")
        st.stop()

# RTSP
rtsp_url = ""
if mode == "RTSP / IP Cam":
    rtsp_url = st.sidebar.text_input(
        "RTSP/HTTP(S) 串流網址",
        value="",
        help="例如：rtsp://user:pass@ip:554/Streaming/Channels/101 或 http(s)://ip:port/stream"
    )

# ------------------------------
# Helpers
# ------------------------------
def parse_classes_filter(text: str) -> Optional[List[int]]:
    if not text.strip():
        return None
    try:
        ints = [int(x.strip()) for x in text.split(",") if x.strip() != ""]
        return ints if len(ints) > 0 else None
    except Exception:
        st.warning("類別過濾格式錯誤，請輸入以逗號分隔的整數，例如：0,1,2。將忽略過濾。")
        return None

@st.cache_resource(show_spinner=True)
def load_model_local_or_upload(model_path: Optional[str], conf: float, iou: float, uploaded_file) -> YOLO:
    if uploaded_file is not None:
        try:
            tmp_path = "uploaded_model.pt"
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            m = YOLO(tmp_path)
            m.overrides["conf"], m.overrides["iou"] = conf, iou
            return m
        except Exception as e:
            st.warning(f"上傳模型載入失敗：{e}，改用其他來源。")
    if model_path:
        try:
            m = YOLO(model_path)
            m.overrides["conf"], m.overrides["iou"] = conf, iou
            return m
        except Exception as e:
            st.warning(f"指定模型路徑載入失敗：{e}，改用回退 yolov8n.pt。")
    m = YOLO("yolov8n.pt")
    m.overrides["conf"], m.overrides["iou"] = conf, iou
    return m

@st.cache_resource(show_spinner=True)
def load_model_from_hf(repo_id: str, filename: str, revision: Optional[str], conf: float, iou: float) -> YOLO:
    hf_token = os.environ.get("HF_TOKEN_FOR_STREAMLIT", None)
    if hf_token is None:
        try:
            hf_token = st.secrets.get("HF_TOKEN_FOR_STREAMLIT", None)
        except Exception:
            pass
    try:
        cached_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=(revision or None),
            token=hf_token,          # 私有/受保護 repo 需要 token
            local_dir="hf_models",
            local_dir_use_symlinks=False
        )
    except Exception as e:
        st.error(f"Hugging Face 下載失敗：{e}\n請確認 repo_id/檔名/權限，以及 HF_TOKEN_FOR_STREAMLIT 是否正確設定。")
        st.stop()
    try:
        m = YOLO(cached_path)
        m.overrides["conf"], m.overrides["iou"] = conf, iou
        return m
    except Exception as e:
        st.error(f"從 Hugging Face 下載到的模型載入失敗：{e}")
        st.stop()

def get_model_and_names_info(m: YOLO):
    try:
        model_name = getattr(m, "ckpt_path", None) or getattr(m, "model", None) or "Unknown"
    except Exception:
        model_name = "Unknown"
    try:
        names = getattr(m, "names", None)
        if isinstance(names, dict):
            names = [names[i] for i in sorted(names.keys())]
    except Exception:
        names = None
    return model_name, names

def put_text(img, text, org, color=(0, 255, 0)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

# ------------------------------
# Load Model（只在這裡載一次）
# ------------------------------
allowed_classes = parse_classes_filter(classes_text)
if model_source == "本地/上傳 .pt":
    model = load_model_local_or_upload(model_path_input.strip(), conf_thres, iou_thres, uploaded_model)
else:
    model = load_model_from_hf(hf_repo_id.strip(), hf_filename.strip(), hf_revision.strip() or None, conf_thres, iou_thres)

# 顯示模型資訊
model_name, class_names = get_model_and_names_info(model)
with st.sidebar.expander("🧩 模型與類別資訊", expanded=True):
    st.write("**目前載入模型**：", str(model_name))
    if class_names:
        st.write("**類別清單（names）**：")
        st.code(", ".join(map(str, class_names)), language="text")
    else:
        st.warning("讀不到類別名稱（names）。若是 yolov8n.pt，沒有你的自訂類別。")

# ------------------------------
# System Info
# ------------------------------
try:
    import torch
    gpu_msg = "✅ 使用 GPU" if torch.cuda.is_available() else "💻 使用 CPU"
except Exception:
    gpu_msg = "💻 使用 CPU（未偵測到 torch.cuda）"
st.sidebar.markdown("---")
st.sidebar.write("🖥️ 執行環境：", gpu_msg)

# ------------------------------
# Inference Utility
# ------------------------------
def run_inference_on_frame(bgr: np.ndarray) -> np.ndarray:
    try:
        results = model.predict(
            source=bgr,
            conf=conf_thres,
            iou=iou_thres,
            classes=allowed_classes,
            verbose=False,
            imgsz=640
        )
        annotated = results[0].plot()  # BGR
        try:
            det_count = 0
            if hasattr(results[0], "boxes") and results[0].boxes is not None and results[0].boxes.data is not None:
                det_count = int(results[0].boxes.data.shape[0])
            put_text(annotated, f"Detections: {det_count}", (10, 60), (0, 255, 255))
        except Exception:
            pass
        return annotated
    except Exception as e:
        st.warning(f"推論時發生錯誤：{e}")
        return bgr

# ------------------------------
# Mode 1: WebRTC（手機後鏡頭）
# ------------------------------
if mode == "WebRTC（手機後鏡頭）":
    st.markdown("### 📱 即時鏡頭（手機後鏡頭 environment）")
    st.write("開啟手機瀏覽器進入此頁面，允許相機權限即可使用後鏡頭。")

    class VideoProcessor:
        def __init__(self):
            self.last_time = time.time()
            self.fps = 0.0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            out = run_inference_on_frame(img)
            now = time.time()
            dt = now - self.last_time
            self.last_time = now
            if dt > 0:
                cur_fps = 1.0 / dt
                self.fps = 0.9 * self.fps + 0.1 * cur_fps if self.fps > 0 else cur_fps
            put_text(out, f"FPS: {self.fps:.1f}", (10, 30), (0, 255, 0))
            return av.VideoFrame.from_ndarray(out, format="bgr24")

    webrtc_streamer(
        key="yolo-homecage-hf-safeload",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": {"facingMode": {"exact": "environment"}}, "audio": False},
        async_processing=True,
    )

# ------------------------------
# Mode 2: RTSP / IP Cam
# ------------------------------
elif mode == "RTSP / IP Cam":
    st.markdown("### 🌐 RTSP / IP Cam 串流偵測")
    placeholder = st.empty()
    col1, col2 = st.columns([1,1])
    start_btn = col1.button("▶️ 開始串流")
    stop_btn = col2.button("⏹ 停止")

    if "rtsp_running" not in st.session_state:
        st.session_state.rtsp_running = False
    if "rtsp_cap" not in st.session_state:
        st.session_state.rtsp_cap = None

    if start_btn and rtsp_url.strip():
        st.session_state.rtsp_running = True
        if st.session_state.rtsp_cap is not None:
            try: st.session_state.rtsp_cap.release()
            except Exception: pass
        st.session_state.rtsp_cap = cv2.VideoCapture(rtsp_url.strip())
        if not st.session_state.rtsp_cap.isOpened():
            st.error("無法開啟串流，請確認 URL 與憑證/網路是否正確。")
            st.session_state.rtsp_running = False

    if stop_btn:
        st.session_state.rtsp_running = False
        if st.session_state.rtsp_cap is not None:
            try: st.session_state.rtsp_cap.release()
            except Exception: pass
        st.session_state.rtsp_cap = None

    if st.session_state.rtsp_running and st.session_state.rtsp_cap is not None:
        fps_avg, last_time = 0.0, time.time()
        while st.session_state.rtsp_running:
            ret, frame = st.session_state.rtsp_cap.read()
            if not ret:
                st.warning("讀不到影像，稍後再試或檢查串流來源。")
                break
            out = run_inference_on_frame(frame)
            now = time.time()
            dt = now - last_time
            last_time = now
            cur_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_avg = 0.9 * fps_avg + 0.1 * cur_fps if fps_avg > 0 else cur_fps
            put_text(out, f"FPS: {fps_avg:.1f}", (10, 30), (0, 255, 0))
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb, channels="RGB", use_container_width=True)
            time.sleep(0.001)

# ------------------------------
# Mode 3: 單張圖片上傳
# ------------------------------
elif mode == "單張圖片上傳測試":
    st.markdown("### 🖼️ 單張圖片上傳測試")
    img_file = st.file_uploader("選擇圖片", type=["jpg", "jpeg", "png", "bmp"])
    if img_file is not None:
        file_bytes = np.frombuffer(img_file.read(), np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("讀取圖片失敗，請換一個檔案再試。")
        else:
            out = run_inference_on_frame(bgr)
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            st.image(rgb, caption="推論結果", use_container_width=True)

# ------------------------------
# Mode 4: 伺服器側示範影片（循環播放）
# ------------------------------
elif mode == "示範影片（伺服器側）":
    st.markdown("### 🎞️ 示範影片（伺服器端循環播放並標注）")
    demo_path = st.text_input("示範影片檔名/路徑", value="demo_mouse.mp4", help="將 MP4 放在 app 同層，或填入完整路徑")
    placeholder = st.empty()
    col1, col2 = st.columns([1,1])
    start_btn = col1.button("▶️ 開始播放")
    stop_btn = col2.button("⏹ 停止")

    if "demo_running" not in st.session_state:
        st.session_state.demo_running = False
    if "demo_cap" not in st.session_state:
        st.session_state.demo_cap = None

    if start_btn:
        if not os.path.exists(demo_path):
            st.error(f"找不到示範影片：{demo_path}")
        else:
            st.session_state.demo_running = True
            if st.session_state.demo_cap is not None:
                try: st.session_state.demo_cap.release()
                except Exception: pass
            st.session_state.demo_cap = cv2.VideoCapture(demo_path)
            if not st.session_state.demo_cap.isOpened():
                st.error("無法開啟示範影片。")
                st.session_state.demo_running = False

    if stop_btn:
        st.session_state.demo_running = False
        if st.session_state.demo_cap is not None:
            try: st.session_state.demo_cap.release()
            except Exception: pass
        st.session_state.demo_cap = None

    if st.session_state.demo_running and st.session_state.demo_cap is not None:
        fps_avg, last_time = 0.0, time.time()
        while st.session_state.demo_running:
            ret, frame = st.session_state.demo_cap.read()
            if not ret:
                st.session_state.demo_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            out = run_inference_on_frame(frame)
            now = time.time()
            dt = now - last_time
            last_time = now
            cur_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_avg = 0.9 * fps_avg + 0.1 * cur_fps if fps_avg > 0 else cur_fps
            put_text(out, f"FPS: {fps_avg:.1f}", (10, 30), (0, 255, 0))
            rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            placeholder.image(rgb, channels="RGB", use_container_width=True)
            time.sleep(0.001)

# ------------------------------
# Footer / Tips
# ------------------------------
with st.expander("🔎 疑難排解"):
    st.markdown(
        """
- **全無標註**：先看側欄「🧩 模型與類別資訊」是否載到你的自訂模型與類別；若顯示 yolov8n.pt，代表未載入自家權重。
- **Hugging Face 私有 repo**：請在 Environment 設定 `HF_TOKEN_FOR_STREAMLIT`。
- **沒有框**：請先清空類別過濾、把 conf 降至 0.15~0.25、IoU 0.45~0.55，再逐步調回。
- **RTSP**：手機瀏覽器通常不支援 RTSP；用「示範影片（伺服器側）」展示體驗。
        """
    )
