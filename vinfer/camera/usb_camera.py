# vinfer/usb_camera.py
import cv2
import threading
import queue
import time
import signal
from ..utils import logger

class USBCamera:
    """Singleton class for USB camera operations (one handle per device)"""
    _instance = None
    _lock = threading.Lock()  # Thread lock for singleton safety

    def __new__(cls, dev_id=0, width=1280, height=720, fps=30):
        """Create singleton instance (only one per device ID)"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init_camera(dev_id, width, height, fps)
                    cls._instance.ref_count = 0
                    cls._instance._preview_window_name = f"USB_Preview_{dev_id}"
                    cls._instance._preview_thread = None
                    cls._instance._preview_running = False

        cls._instance.ref_count += 1
        logger.info(f"USB camera ref count increased: {cls._instance.ref_count} (dev: {cls._instance.dev_id})")
        return cls._instance

    def _init_camera(self, dev_id, width, height, fps):
        """Initialize USB camera with V4L2 backend (avoid GStreamer conflicts)"""
        self.dev_id = dev_id
        self.frame_queue = queue.Queue(maxsize=2)  # Shared queue for preview/inference
        self.exit_flag = threading.Event()        # Exit signal for graceful shutdown
        self.cap = None                           # Camera capture handle
        self.read_thread = None                   # Frame reading thread

        # Force V4L2 backend to avoid GStreamer resource conflicts
        self.cap = cv2.VideoCapture(dev_id, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            logger.error(f"USB camera initialization failed: /dev/video{dev_id}")
            raise RuntimeError(f"Failed to open USB camera: /dev/video{dev_id}")

        # Configure camera parameters (only once at initialization)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer to reduce conflicts
        logger.info(f"USB camera initialized successfully: /dev/video{dev_id} (resolution: {width}x{height}, FPS: {fps})")

    def _read_frames(self):
        """Background thread to read frames continuously (only one reader thread)"""
        while not self.exit_flag.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from USB camera, retrying...")
                time.sleep(0.1)
                continue
            # Overwrite old frames to ensure real-time performance
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def start_read(self):
        """Start frame reading thread (only start once)"""
        if self.read_thread is None or not self.read_thread.is_alive():
            self.exit_flag.clear()
            self.read_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.read_thread.start()
            logger.info("USB camera frame reading thread started")

    def get_frame(self):
        """Get latest frame from shared queue (for preview/inference)"""
        try:
            return self.frame_queue.get(timeout=0.5)
        except queue.Empty:
            logger.warning("USB camera frame queue is empty, no frames available")
            return None

    def _preview_worker(self, preview_size):
        cv2.namedWindow(self._preview_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._preview_window_name, preview_size[0], preview_size[1])
        logger.info(f"USB preview window created: {self._preview_window_name} (thread: {threading.current_thread().name})")
	        
        self._preview_running = True
        # while self._preview_running and not self.exit_flag.is_set():
        while self._preview_running:
            frame = self.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            preview_frame = cv2.resize(frame, preview_size)
            cv2.imshow(self._preview_window_name, preview_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # Q/ESC
               logger.info(f"USB preview stopped by user (Q/ESC): {self._preview_window_name}")
               break

            if cv2.getWindowProperty(self._preview_window_name, cv2.WND_PROP_VISIBLE) < 1:
               logger.info(f"USB preview window closed by user (X): {self._preview_window_name}")
               break
        
        self._preview_running = False
        try:
            cv2.destroyWindow(self._preview_window_name)
            time.sleep(0.05)  
            logger.info(f"USB preview window destroyed: {self._preview_window_name} (thread: {threading.current_thread().name})")
        except Exception as e:
            logger.warning(f"Failed to destroy preview window: {e}")

    def start_preview(self, preview_size=(480, 360)):
        if self._preview_thread is None or not self._preview_thread.is_alive():
            self._preview_thread = threading.Thread(
                target=self._preview_worker,
                args=(preview_size,),
                daemon=True,
                name=f"USB_Preview_{self.dev_id}"
            )
            self._preview_thread.start()
            logger.info(f"USB preview started: /dev/video{self.dev_id} (size: {preview_size})")
        return self._preview_thread

    def stop_preview(self):            
        self._preview_running = False
        if self._preview_thread is not None and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=2)
            logger.info(f"USB preview stopped: /dev/video{self.dev_id}")

    '''
    def preview(self, preview_size=(640, 480), window_name="USB Preview"):
        """
        Preview thread logic (q key closes window only, no resource release)
        :param preview_size: (width, height) for preview window
        :param window_name: preview window name
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, preview_size[0], preview_size[1])
        
        while True:
            frame = self.get_frame()
            if frame is None:
                continue
            preview_frame = cv2.resize(frame, preview_size)
            cv2.imshow(window_name, preview_frame)
            
            # Key event handling: q/ESC closes window only (no stop)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # q or ESC
                logger.info(f"USB preview window closed (dev: {self.dev_id})")
                break
            # Close window by clicking X (OS event)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                logger.info(f"USB preview window closed by user (dev: {self.dev_id})")
                break
        
        # Only destroy window, DO NOT call stop()
        cv2.destroyWindow(window_name)
        # Release reference when preview exits
        self.release_ref()
    '''
    def release_ref(self):
        """Decrease ref count, release resource only when ref count == 0"""
        with self._lock:
            self.ref_count -= 1
            logger.info(f"USB camera ref count decreased: {self.ref_count} (dev: {self.dev_id})")
            # Only stop and release when no consumer is using
            if self.ref_count <= 0:
                self._stop_and_release()
                # Reset singleton
                USBCamera._instance = None
                logger.info(f"USB camera fully released: /dev/video{self.dev_id}")

    def _stop_and_release(self):
        """Real resource release logic (internal use only)"""
        self.stop_preview()

        # Stop frame reading thread
        self.exit_flag.set()
        if self.read_thread is not None and self.read_thread.is_alive():
            self.read_thread.join(timeout=1)
        # Release camera handle
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            logger.info(f"USB camera handle released: /dev/video{self.dev_id}")
            self.cap = None
        # Clear frame queue
        while not self.frame_queue.empty():
            self.frame_queue.get_nowait()
        logger.info(f"USB camera resources released: /dev/video{self.dev_id}")

    # Expose stop method for manual full release (optional)
    def stop(self):
        """Manual stop (force release all resources)"""
        self.ref_count = 0  # Force ref count to 0
        self.release_ref()

# Public interface for simplified usage
def init_usb_camera(dev_id=0, width=1280, height=720, fps=30):
    """Initialize USB camera singleton and start frame reading thread"""
    try:
        camera = USBCamera(dev_id, width, height, fps)
        camera.start_read()
        return camera
    except RuntimeError as e:
        logger.error(e)
        return None

# Public interface for preview
def start_usb_preview(dev_id=0, preview_size=(480, 360)):
    """
    Start USB camera preview thread (consumes shared frames from singleton instance)
    :param dev_id: USB camera device ID (default: 0)
    :param preview_size: Tuple (width, height) for preview window (default: (480, 360))
    :return: Preview thread handle (for join/control)
    :raises RuntimeError: If camera initialization fails
    """
    try:
        # Get singleton camera instance (initialize if not exists)
        camera = USBCamera(dev_id)
        # Ensure frame reading thread is running (critical for preview)
        camera.start_read()
        logger.info(f"USB camera preview thread started (device ID: {dev_id}, preview size: {preview_size})")
        return camera.start_preview(preview_size)
    except RuntimeError as e:
        logger.error(f"Failed to start USB preview: {e}")
        raise  # Re-raise error to let caller handle it
