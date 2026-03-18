import cv2
import re
import numpy as np
import os
import signal
import subprocess
import logging
from logging.handlers import RotatingFileHandler
from .constants import FFMPEG_PIDS
from .resolution_cache import get_rtsp_resolution

def setup_logger():
    log_dir = os.path.expanduser("~/.vinfer")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "vinfer.log")
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  
        backupCount=5,         
        encoding="utf-8"
    )
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler.setFormatter(logging.Formatter(log_format))
    
    handlers = [logging.StreamHandler(), file_handler]
    logging.basicConfig(level=logging.WARNING, handlers=handlers)
    # logging.basicConfig(level=logging.INFO, handlers=handlers)

setup_logger()
logger = logging.getLogger("vinfer")


def get_usb_frame(dev_id):
    try:
        cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
        backend = cap.getBackendName()
        logger.info(f"USB camera using backend: {backend} (Target: V4L2)")
    except Exception as e:
        logger.warning(f"V4L2 backend is unavailable, falling back to the default backend:{e}")
        cap = cv2.VideoCapture(dev_id)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def is_valid_frame(frame):
    return frame is not None and frame.size > 0

def kill_all_ffmpeg():
    global FFMPEG_PIDS
    try:
        for pid in FFMPEG_PIDS:
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except:
                pass
        FFMPEG_PIDS = []
    except:
        pass

def extract_frame_stable(stream_url, stream_type, video_duration=0):
    global vod_current_offset, vod_step
    max_attempts = 3
    rtsp_width, rtsp_height = (0, 0)
    if stream_type == "rtsp":
        rtsp_width, rtsp_height = get_rtsp_resolution(stream_url)

    for attempt in range(max_attempts):
        kill_all_ffmpeg()
        try:
            if stream_type == "live" or stream_type == "rtsp":
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-loglevel', 'error',
                    '-rtsp_transport', 'tcp',
                    '-stimeout', '10000000',
                    '-max_delay', '1000000',
                    '-i', stream_url,
                    '-ss', '0.5',
                    '-vframes', '1',
                    '-f', 'image2pipe',
                    '-vcodec', 'mjpeg',
                    '-'
                ]

            elif stream_type == "vod":
                if video_duration > 0 and vod_current_offset > video_duration - 5:
                    print(f"⚠️ VOD position exceeds duration, reset to {vod_start_offset} seconds")
                    vod_current_offset = vod_start_offset
                
                ffmpeg_cmd = [
                    'ffmpeg',
                    '-timeout', '8000000',
                    '-ss', str(vod_current_offset),
                    '-i', stream_url,
                    '-vframes', '1',
                    '-s', '480x360',
                    '-f', 'image2pipe',
                    '-vcodec', 'mjpeg',
                    '-hide_banner',
                    '-loglevel', 'error',
                    '-'
                ]
                if args.debug:
                    print(f"📌 VOD frame extraction position: {vod_current_offset} seconds")

            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid
            )
            FFMPEG_PIDS.append(proc.pid)

            frame_data = proc.stdout.read()
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            proc.wait(timeout=20 if stream_type == "rtsp" else 8)
            if proc.pid in FFMPEG_PIDS:
                FFMPEG_PIDS.remove(proc.pid)

            if frame is not None and is_valid_frame(frame):
                if stream_type == "vod":
                    vod_current_offset += vod_step
                    print(f"📌 VOD position advanced: {vod_current_offset} seconds (step {vod_step} seconds)")
                
                target_size = (480, 360)
                if frame.shape[1] != target_size[0] or frame.shape[0] != target_size[1]:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                
                return frame

            time.sleep(1.0)
            
        except subprocess.TimeoutExpired:
            kill_all_ffmpeg()
            print(f"⚠️ Frame extraction timeout (attempt {attempt+1})")
        except Exception as e:
            kill_all_ffmpeg()
            print(f"⚠️ Frame extraction exception (attempt {attempt+1}): {e}")
        time.sleep(1.0)
    
    kill_all_ffmpeg()
    raise Exception(f"Frame extraction failed (retried {max_attempts} times)")

def check_usb_camera(device_path: str) -> bool:
    if not os.path.exists(device_path):
        return False
    return True

def init_shared_camera(dev_id):
    cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
    ret = cap.isOpened()
    cap.release()
    return ret

def detect_language(text: str) -> str:

    clean_text = re.sub(r'[^\w\s]', '', text).strip()
    if not clean_text:
        return "unknown"
    
    chinese_chars = re.findall(r'[\u4e00-\u9fff]', clean_text)
    chinese_count = len(chinese_chars)
    
    english_chars = re.findall(r'[a-zA-Z]', clean_text)
    english_count = len(english_chars)
    
    total = chinese_count + english_count
    if total == 0:
        return "unknown"
    
    chinese_ratio = chinese_count / total
    english_ratio = english_count / total
    
    if chinese_ratio > english_ratio:
        return "zh"
    elif english_ratio > chinese_ratio:
        return "en"
    else:
        first_char = next((c for c in clean_text if c.strip()), '')
        if re.match(r'[\u4e00-\u9fff]', first_char):
            return "zh"
        else:
            return "en"