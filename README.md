# VisionInfer
Lightweight Visual Language Model (VLM) Inference Tool optimized for **Jetson Edge Devices** and x86 platforms. Supports real-time inference for USB/RTSP cameras, VOD videos, and live streams with motion detection, frame deduplication, and efficient resource management.

## Features
- 🎥 Multi-source support: USB cameras, RTSP streams, VOD files, live network streams
- 🚀 Motion-gated inference (only run inference when motion detected)
- 🎯 Frame deduplication (skip similar frames via L2 feature comparison)
- 📊 Real-time performance monitoring (encoding/inference time, frame metrics)
- 🔧 Jetson-optimized: Tailored for ARM64 architecture and limited edge resources
- 🎛️ Configurable parameters: Compression quality, inference interval, motion threshold
- 🪵 Debug mode for troubleshooting (--debug flag)

## Requirements
### General Requirements
- Python 3.8+
- OpenCV (cv2)
- NumPy
- psutil
- Ollama (v0.1.40+)
- FFmpeg (for frame extraction from streams/files)

### Jetson-Specific Requirements
- Jetson Nano/Xavier NX/Orin (JetPack 5.0+)
- Minimum 8GB RAM 

## Installation
### 1. Base Dependencies (All Platforms)
```bash
# Install system dependencies
sudo apt update && sudo apt install -y ffmpeg python3-pip pipx
```
### 2. Install VisionInfer
#### 2.1 For Jetson (Pre-installed System OpenCV)
To avoid breaking system dependencies (e.g., JetPack's pre-built OpenCV), use --system-site-packages to reuse the system's OpenCV:
```bash
pipx install --system-site-packages vinfer
```
#### 2.2 For Other Systems (No Special OpenCV)
Install with full dependencies (includes OpenCV) if your system doesn't have a pre-configured OpenCV:
```bash
pipx install vinfer[full]
```

### 3. Ollama Installation

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 4. Jetson Resource Configuration (Optional)
#### 4.1 Increase Swap Space
```bash
# Create 4GB swap file
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Make swap permanent (survive reboot)
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

#### 4.2 Configure GPU Memory (Jetson Orin/Nano)
```bash
# For Jetson Orin (set 16GB GPU memory)
sudo nvpmodel -m 0
sudo jetson_clocks

# For Jetson Nano (set max performance mode)
sudo nvpmodel -m 0
sudo jetson_clocks
```

#### 4.3 Pull Optimized Model (Jetson)
```bash
# Recommended lightweight model for Jetson
ollama pull qwen3.5:2b
```

## Quick Start
### 1. USB Camera Inference
```bash
# Basic USB camera (device ID 0) with debug logs
vinfer cam --usb-dev 0 --debug

# USB camera with motion detection (infer only on motion)
vinfer cam --usb-dev 0 --motion-gate --motion-threshold 500

# USB camera with frame deduplication (skip similar frames)
vinfer cam --usb-dev 0 --dedup --interval 2.0
```

### 2. RTSP Camera Inference
```bash
# Basic RTSP stream (default credentials)
vinfer cam --rtsp-host 192.168.1.10 --rtsp-user admin --rtsp-pass password --debug

# RTSP with custom compression (320x240) and JPG quality (80)
vinfer --rtsp-host 192.168.1.10 --compress-size 320x240 --jpg-quality 80
```

### 3. VOD (Video File) Analysis
```bash
# Local video file (analyze every 30 frames)
vinfer analyze --type vod --file /path/to/video.mp4 --start 0 --step 30

# Network VOD URL (e.g., MP4 stream)
vinfer analyze --type vod --url https://example.com/video.mp4 --debug
```

### 4. Live Stream Analysis
```bash
# HLS live stream (e.g., .m3u8)
vinfer analyze --type live --url https://example.com/stream.m3u8 --interval 1.0
```

## Command Reference
### Core Subcommands
| Subcommand | Description |
|------------|-------------|
| `cam`      | Real-time camera inference (USB/RTSP) |
| `analyze`  | Offline video/live stream analysis |

### Common Arguments
| Argument | Short | Description | Default |
|----------|-------|-------------|---------|
| `--model` | `-m`  | Ollama model name | `qwen3.5:2b` |
| `--compress-size` | `-s` | Frame compression resolution (WxH) | `480x360` |
| `--jpg-quality` | `-q` | JPG compression quality (0-100) | `70` |
| `--motion-gate` | `-g` | Enable motion detection (infer only on motion) | `False` |
| `--motion-threshold` | `-T` | Minimum motion area (pixels) | `500` |
| `--dedup` | `-D` | Enable frame deduplication (disabled if motion-gate is on) | `False` |
| `--interval` | `-i` | Inference interval (seconds/frame) | `1.0` |
| `--debug` | `-d` | Enable verbose debug logging | `False` |

### Cam Subcommand Arguments
| Argument | Short | Description |
|----------|-------|-------------|
| `--rtsp-host` | `-H` | RTSP server IP/domain (enables RTSP mode) |
| `--rtsp-user` | `-U` | RTSP authentication username | `admin` |
| `--rtsp-pass` | `-P` | RTSP authentication password | `""` |
| `--usb-dev` | `-u` | USB camera device ID (0 = /dev/video0) | `0` |
| `--show-preview` | `-p` | Start live preview window | `False` |

### Analyze Subcommand Arguments
| Argument | Short | Description |
|----------|-------|-------------|
| `--type` | `-t` | Analysis type (`vod`/`live`) | **Required** |
| `--file` | `-f` | Local VOD file path |
| `--url` | `-u` | Network VOD/live stream URL |
| `--start` | `-st` | Start frame number (0-based) | `0` |
| `--step` | `-sp` | Inference frame interval | `1` |

## Troubleshooting
### Common Issues & Solutions
#### 1. EOF Error During Frame Extraction
- **Symptom**: `EOFError`/`IOError` when reading frames from RTSP/live streams
- **Solutions**:
  - Increase RTSP timeout: Add `-stimeout 20000000` to FFmpeg command (code already includes this)
  - Check network stability (RTSP streams require low latency)
  - Use TCP for RTSP: `--rtsp-transport tcp` (enabled by default in code)

#### 2. Zombie Processes (FFmpeg/Ollama)
- **Symptom**: Orphaned FFmpeg/Ollama processes consuming resources
- **Solutions**:
  - The code includes `kill_all_ffmpeg()` and `stop_ollama_serve()` for cleanup
  - Manually kill zombie processes:
    ```bash
    # Kill all FFmpeg processes
    sudo pkill -f ffmpeg
    
    # Restart Ollama service
    sudo systemctl restart ollama
    ```

#### 3. Resource Exhaustion (Jetson)
- **Symptom**: `Out of memory` errors or slow inference
- **Solutions**:
  - Use smaller models (qwen3.5:2b instead of 7b)
  - Increase swap space (see Installation > Jetson Configuration)
  - Reduce frame resolution (`--compress-size 320x240`)
  - Increase inference interval (`--interval 2.0` or higher)

#### 4. Frame Extraction Failure
- **Symptom**: `Frame extraction failed, unable to perform inference`
- **Solutions**:
  - Verify RTSP URL/USB device accessibility
  - Check FFmpeg installation (`ffmpeg -version`)
  - For RTSP: Ensure camera is online and credentials are correct

#### 5. Continuous Inference Errors
- **Symptom**: `Continuous inference exception: [error message]`
- **Solutions**:
  - Enable debug mode (`--debug`) to see detailed error logs
  - Check Ollama service status (`sudo systemctl status ollama`)
  - Verify model is pulled (`ollama list` to check installed models)

## Known Limitations
### Jetson-Specific Limitations
- **Model Size**: Avoid 7B+ models (e.g., qwen3.5:7b) on Jetson Nano/Xavier NX—use `qwen3.5:2b` for stable performance
- **Inference Speed**: 2B models run at ~1-2 FPS on Jetson Orin, ~0.5 FPS on Jetson Nano
- **Preview Window**: May be slow on Jetson Nano (disable with `--no-preview` if needed)

### General Limitations
- **RTSP Latency**: RTSP streams may have 1-3s latency (normal for TCP transport)
- **Frame Deduplication**: May skip valid frames in low-motion scenarios (adjust `DEDUP_THRESHOLD` if needed)
- **Motion Detection**: Sensitive to lighting changes (tune `--motion-threshold` for your environment)

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- [Ollama](https://ollama.com/) for lightweight LLM inference
- [OpenCV](https://opencv.org/) for computer vision processing
- [NVIDIA Jetson](https://developer.nvidia.com/jetson) for edge AI platform support
