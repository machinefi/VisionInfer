import sys
import os
import cv2
import threading
import time
import signal
import atexit
import argparse
import queue
import vinfer

from .constants import (
    EXIT_FLAG, input_thread, preview_stop_event, preview_thread_handle,
    FRAME_INTERVAL, DEDUP_THRESHOLD, MOTION_THRESHOLD, input_queue,
    FRAME_QUEUE, FRAME_INFO_QUEUE, DEFAULT_PROMPT
)
from .cli import add_common_arguments
from .backend.ollama_manager import start_ollama_serve, stop_ollama_serve
from .frame_producer import start_frame_producer, stop_frame_producer
from .input_listener import input_listener, preview_thread
from .utils import logger, check_usb_camera, init_shared_camera, extract_frame_stable, kill_all_ffmpeg
from .inference_core import infer_frame
from .camera.usb_camera import init_usb_camera, start_usb_preview, USBCamera
from .camera.rtsp_camera import init_rtsp_camera, start_rtsp_preview, RTSPCamera

# Disable Jetson font warnings (adapted for screen preview window)
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
# os.environ["QT_FONT_DPI"] = "96"
# os.environ["QT_LOGGING_RULES"] = "qt.fontdatabase.warning=false"


def signal_handler(sig, frame):
    global EXIT_FLAG, input_thread
    print("\n🛑 Received exit signal...")
    EXIT_FLAG = True
    if input_thread and input_thread.is_alive():
        sys.stdin.close()
        input_thread.join(timeout=1)
    stop_ollama_serve() 
    sys.exit(0)

def main():
    global EXIT_FLAG, input_thread, preview_stop_event, preview_thread_handle, FRAME_INTERVAL, DEDUP_THRESHOLD, MOTION_THRESHOLD

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(stop_ollama_serve)

    # Argument parser
    parser = argparse.ArgumentParser(
        prog="vinfer",
        description="VisionInfer - Lightweight VLM Inference Tool (supports camera/VOD/live stream)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Version
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"vinfer {vinfer.__version__}", 
        help="Show vinfer version and exit"
    )
    
    # Subcommand parser
    subparsers = parser.add_subparsers(
        dest="subcommand",
        required=True,
        help="Subcommands: cam (real-time camera) / analyze (offline analysis)"
    )

    # Subcommand 1: vinfer cam (real-time camera)
    parser_cam = subparsers.add_parser(
        "cam",
        help="Real-time camera inference (supports RTSP/USB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # RTSP/USB mutually exclusive group
    cam_group = parser_cam.add_mutually_exclusive_group(required=False)
    cam_group.add_argument(
        "--rtsp-host", "-H",
        type=str,
        help="RTSP server IP/domain (e.g., 192.168.1.10) - enables RTSP mode"
    )
    cam_group.add_argument(
        "--usb-dev", "-u",
        type=int,
        default=0,
        help="USB camera device ID (0 for /dev/video0)"
    )
    
    # RTSP optional parameters
    parser_cam.add_argument(
        "--rtsp-user", "-U",
        type=str,
        default="admin",
        help="RTSP authentication username"
    )
    parser_cam.add_argument(
        "--rtsp-pass", "-P",
        type=str,
        default="",
        help="RTSP authentication password"
    )
    parser_cam.add_argument(
        "--rtsp-channel", "-C",
        type=int,
        default=1,
        help="RTSP channel number"
    )
    parser_cam.add_argument(
        "--rtsp-stream", "-S",
        type=int,
        default=0,
        help="RTSP stream number"
    )
    parser_cam.add_argument(
        "--show-preview", "-p", 
        action='store_true',
        help='Start preview window'
    )
    
    # Add common arguments
    add_common_arguments(parser_cam)

    # Subcommand 2: vinfer analyze (offline analysis)
    parser_analyze = subparsers.add_parser(
        "analyze",
        help="Offline video analysis (supports VOD/live stream)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Core type parameter (required)
    parser_analyze.add_argument(
        "--type", "-t",
        type=str,
        required=True,
        choices=["vod", "live"],
        help="Analysis type: vod (video on demand) / live (live stream)"
    )
    
    # VOD/Live core parameters (mutually exclusive)
    analyze_group = parser_analyze.add_mutually_exclusive_group(required=False)
    analyze_group.add_argument(
        "--file", "-f",
        type=str,
        help="Local VOD video file path (e.g., /home/videos/test.mp4)"
    )
    analyze_group.add_argument(
        "--url", "-u",
        type=str,
        help="Network VOD URL/live stream URL (e.g., https://example.com/stream.m3u8)"
    )
    
    # Frame control parameters
    parser_analyze.add_argument(
        "--start", "-st",
        type=int,
        default=0,
        help="Start inference frame number (0-based)"
    )
    parser_analyze.add_argument(
        "--step", "-sp",
        type=int,
        default=1,
        help="Inference frame interval (infer every N frames)"
    )
    
    # Add common arguments
    add_common_arguments(parser_analyze)

    # Parse arguments
    args = parser.parse_args()
    # Argument validation
    if args.subcommand == "cam":
        args.continuous = True
        # Auto-generate RTSP URL if RTSP mode enabled
        if args.rtsp_host:
            rtsp_url = f"rtsp://{args.rtsp_host}:554/user={args.rtsp_user}&password={args.rtsp_pass}&channel={args.rtsp_channel}&stream={args.rtsp_stream}"
            if args.debug:
                print(f"📌 Auto-generated RTSP URL: {rtsp_url}")
            args.source_url = rtsp_url
            args.source_type = "rtsp"
        else:
            args.source_dev = f"/dev/video{args.usb_dev}"
            args.source_type = "usb"
    elif args.subcommand == "analyze":
        args.continuous = False
        if args.type == "vod":
            if not args.file and not args.url:
                print("❌ VOD mode requires --file (local) or --url (network)")
                sys.exit(1)
            args.source_type = "vod"
        elif args.type == "live":
            if not args.url:
                print("❌ Live mode requires --url (live stream URL)")
                sys.exit(1)
            args.source_type = "live"
        # Validate frame control parameters
        if args.step <= 0:
            print("❌ --step must be positive integer")
            sys.exit(1)
        args.start_frame = args.start
        args.step_frame = args.step

    # Motion-gate has higher priority than dedup
    if args.motion_gate and args.dedup:
        print(f"📌 Note: motion-gate enabled, dedup automatically disabled (higher priority)")
        args.dedup = False

    # USB camera validation
    if args.source_type == "usb":
        if not check_usb_camera(args.source_dev):
            print("⚠️ USB camera initialization failed, please check camera status")        
            return

    # Start Ollama service
    print("="*60)
    print("🚀 Starting VisionInfer Tool (optimized version)")
    print("="*60)
    if not start_ollama_serve():
        print("❌ Program exited")
        return

    # Initialize preview
    preview_width, preview_height = 480, 360
    if args.show_preview:
        if args.source_type == "usb":
            preview_thread_handle = start_usb_preview(args.usb_dev, preview_size=(480, 360))
            logger.info(f"✅ USB Preview started(ID:{args.usb_dev})")
        elif args.source_type == "rtsp":
            preview_thread_handle = start_rtsp_preview(args.source_url, 0, preview_size=(480, 360))
            logger.info(f"✅ RTSP preview started (URL: {args.source_url[:30]}...)")

    # Start input listener
    input_thread = threading.Thread(target=input_listener, args=(args,), daemon=True)
    input_thread.start()

    # Auto-start continuous inference
    continuous_running = False
    if args.continuous and args.source_type in ['usb', 'rtsp']:
        continuous_running = True
        start_frame_producer(args)
        print(f"\n▶️  Continuous inference started (result interval: {args.interval}s, frame extraction frequency: {FRAME_INTERVAL}s/frame)")

    print(f"Model : {args.model} is loading...")

    # Main thread loop
    try:
        while not EXIT_FLAG:
            try:
                cmd = input_queue.get(timeout=0.5)
                
                # Exit command
                if cmd == "exit":
                    EXIT_FLAG = True
                    continuous_running = False
                    print("📤 Exiting program...")
                    break
                
                # Single inference
                elif cmd == "infer":
                    continuous_running = False
                    stop_frame_producer()
                    frame = None
                    if args.source_type == "usb":
                        frame = get_usb_frame(args.usb_dev)
                    elif args.source_type == "rtsp":
                        frame = extract_frame_stable(args.source_url, "rtsp")

                    if frame is not None:
                        start_total = time.time()
                        _, img_bytes = cv2.imencode('.jpg', frame)
                        raw_image_data = img_bytes.tobytes()
                        img_cost = time.time() - start_total
                        result, infer_cost = infer_frame(args, raw_image_data)
                        total_cost = time.time() - start_total
                        print(f"📝 Inference Result (cost: {infer_cost:.2f}s):")          
                    else:
                        if args.debug:
                            print("❌ Frame extraction failed, skipping inference for this round.") 
                                           
                # Continuous inference control
                elif cmd == "start":
                    if not continuous_running and args.source_type in ['usb', 'rtsp']:
                        continuous_running = True
                        start_frame_producer(args)
                        print(f"▶️  Continuous inference started (interval: {args.interval}s)")
                    else:
                        if not continuous_running:
                            print("⚠️ Continuous inference already running")
                        else:
                            print("⚠️ Continuous inference already running")

                elif cmd == "stop":
                    if continuous_running:
                        continuous_running = False
                        stop_frame_producer()
                        print("⏹️  Continuous inference stopped")
                    else:
                        print("⚠️ Continuous inference is only supported for USB/RTSP.")
                
                else:
                    print(f"❌ Unknown command: {cmd}")
            
            except queue.Empty:
                if continuous_running and not EXIT_FLAG and args.source_type in ['usb', 'rtsp']:
                    try:
                        raw_image_data = None
                        frame_info = None
                        try:
                            raw_image_data = FRAME_QUEUE.get(timeout=0.1)
                            frame_info = FRAME_INFO_QUEUE.get(timeout=0.1)
                        except queue.Empty:
                            if args.debug:
                                print("⚠️ No frames pending inference, waiting for frame extraction...")
                            time.sleep(0.1)
                            continue
                        
                        result, infer_cost = infer_frame(args, raw_image_data)
                        
                        print(f"\n🎯 Continuous inference result: {result}")
                        if args.debug:
                            print(f"⏱️  Encoding time: {frame_info['img_cost']:.2f}s | Inference time: {infer_cost:.2f}s")
                            print(f"🖼️  Frame info: Resolution {frame_info['frame_shape']} | Size {frame_info['img_size_kb']}KB | JPG quality {frame_info['jpg_quality']}")
                            print(f"⌛  Waiting {args.interval} seconds before next inference...")

                        wait_start = time.time()
                        while (time.time() - wait_start) < args.interval and not EXIT_FLAG and continuous_running:
                            time.sleep(0.1)

                    except Exception as e:
                        if not EXIT_FLAG:
                            print(f"\n⚠️ Continuous inference exception: {e}")
                            time.sleep(args.interval)

                continue

            except Exception as e:
                if not EXIT_FLAG:
                    print(f"⚠️ Main loop exception: {e}")
                
    except KeyboardInterrupt:
        EXIT_FLAG = True
        continuous_running = False
        stop_frame_producer()
        if args.debug:
            print("\n⚠️ Program interrupted by user")
    finally:
        # Cleanup
        EXIT_FLAG = True
        stop_frame_producer()
        if preview_thread_handle and preview_thread_handle.is_alive():
            preview_stop_event.set()
            preview_thread_handle.join(timeout=2)
        stop_ollama_serve()
        kill_all_ffmpeg()

        if args.source_type == "usb":
            try:
                camera = USBCamera(args.usb_dev)
                camera.stop()
            except Exception as e:
                print(f"⚠️ Failed to force release USB camera: {e}")
        elif args.source_type == "rtsp":
            try:
                camera = RTSPCamera(args.usb_dev)
                camera.stop()
            except Exception as e:
                print(f"⚠️ Failed to force release RTSP camera: {e}")
        else:
            cv2.destroyAllWindows()
            
        print("✅ Program exited cleanly")

if __name__ == "__main__":
    main()