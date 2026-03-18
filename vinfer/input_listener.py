import sys
import time
import threading
import cv2
from .constants import (
    EXIT_FLAG, input_queue, preview_stop_event, preview_thread_handle,
    vod_start_offset, vod_step, vod_current_offset
)
from .backend.ollama_manager import print_ollama_usage, print_ollama_perf
from .utils import kill_all_ffmpeg

def input_listener(args):
    global vod_start_offset

    print("\n📢 Tool started successfully, supported commands:")
    print("   - infer: Single frame inference")
    if args.source_type in ['usb', 'rtsp']:
        print("   - start: Start continuous inference")
        print("   - stop: Stop continuous inference")
    print("   - perf [model name]: Test inference performance")
    print("   - preview on/off: Start/stop preview")
    print("   - exit: Exit program")
    print("="*60)

    while not EXIT_FLAG:
        try:
            import select
            if select.select([sys.stdin], [], [], 0.1)[0]:
                user_input = sys.stdin.readline().strip().lower()
                if not user_input:
                    continue
                
                if user_input in ["usage", "ollama", "stats"]:
                    print_ollama_usage()
                elif user_input.startswith("perf"):
                    parts = user_input.split()
                    model_name = parts[1] if len(parts)>1 else args.model
                    print_ollama_perf(model_name)
                elif user_input.startswith("step "):
                    try:
                        new_step = int(user_input.split()[1])
                        if new_step > 0:
                            global vod_step
                            vod_step = new_step
                            vod_current_offset = vod_start_offset
                            print(f"📌 VOD step set to: {vod_step} seconds (position reset)")
                        else:
                            print("❌ Step must be positive integer")
                    except:
                        print("❌ Command format: step 30")
                elif user_input.startswith("start "):
                    try:
                        new_start = int(user_input.split()[1])
                        if new_start >= 0:
                            vod_start_offset = new_start
                            vod_current_offset = new_start
                            print(f"📌 VOD start position set to: {vod_start_offset} seconds")
                        else:
                            print("❌ Start position must be ≥0")
                    except:
                        print("❌ Command format: start 10")
                elif user_input == "reset":
                    vod_current_offset = vod_start_offset
                    print(f"📌 VOD position reset to: {vod_start_offset} seconds")
                elif user_input == "preview on":
                    if args.source_type in ["usb", "rtsp"]:
                        if preview_thread_handle and preview_thread_handle.is_alive():
                            print("⚠️ Preview window already open")
                        else:
                            preview_stop_event.clear()
                            preview_thread_handle = threading.Thread(
                                target=preview_thread,
                                args=(args.source_type, 
                                      args.source_url if args.source_type == "rtsp" else args.usb_dev, 
                                      (480, 360), 
                                      preview_stop_event),
                                daemon=True
                            )
                            preview_thread_handle.start()
                            print("✅ Preview window started")
                    else:
                        print("❌ Preview only supported for usb/rtsp")
                elif user_input == "preview off":
                    if preview_thread_handle and preview_thread_handle.is_alive():
                        preview_stop_event.set()
                        preview_thread_handle.join(timeout=2)
                        print("✅ Preview window closed")
                    else:
                        print("⚠️ Preview window not open")
                else:
                    input_queue.put(user_input)
        except (EOFError, IOError):
            break
        except Exception as e:
            if not EXIT_FLAG and args.debug:
                print(f"⚠️ Input listener exception: {e}")
    print("✅ Input listener thread exited")

def preview_thread(source_type, source, size, stop_event):
    preview_width, preview_height = size
    cap = None
    if source_type == "usb":
        cap = cv2.VideoCapture(source)
    elif source_type == "rtsp":
        cap = cv2.VideoCapture(source)
    if cap is None or not cap.isOpened():
        print(f"❌ Failed to open {source_type} preview source")
        return
    
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, preview_width)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, preview_height)
    
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (preview_width, preview_height))
        cv2.imshow(f"VisionInfer Preview ({source_type})", frame)
        
        if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
            stop_event.set()
            break

    cap.release()
    cv2.destroyWindow(f"VisionInfer Preview ({source_type})")
