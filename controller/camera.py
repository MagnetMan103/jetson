# python
import cv2
import threading
import time


def create_gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=21,
    flip_method=0,
):
    """
    Return a GStreamer pipeline string for IMX219 / nvarguscamerasrc usage.
    Matches the pipeline builder in `main.py`.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! appsink"
    )


class FrameBuffer:
    """
    Background frame reader that can open:
      - a standard camera index (camera_id), or
      - a GStreamer pipeline string (pipeline) with cv2.CAP_GSTREAMER.

    Usage examples:
      fb = FrameBuffer(camera_id=0)                      # Open default camera
      fb = FrameBuffer(pipeline=create_gstreamer_pipeline())  # Open GStreamer pipeline
    """

    def __init__(
        self,
        camera_id=0,
        pipeline=None,
        use_gstreamer=False,
        buffer_size=1,
        read_sleep=0.005,
        **gst_kwargs,
    ):
        """
        Args:
          camera_id: integer camera id (used if pipeline is None)
          pipeline: explicit GStreamer pipeline string to open
          use_gstreamer: if True and pipeline is None, build pipeline with gst_kwargs
          buffer_size: CV buffer size (set via CAP_PROP_BUFFERSIZE when supported)
          read_sleep: small sleep when read fails to avoid busy-looping
          gst_kwargs: passed to create_gstreamer_pipeline when building a pipeline
        """
        if use_gstreamer and pipeline is None:
            pipeline = create_gstreamer_pipeline(**gst_kwargs)

        if pipeline is not None:
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        else:
            self.cap = cv2.VideoCapture(camera_id)

        # Try to set buffer size (may be ignored on some backends)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, int(buffer_size))
        except Exception:
            pass

        if not self.cap.isOpened():
            raise RuntimeError("Failed to open camera/capture")

        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self._read_sleep = read_sleep

        self.thread = threading.Thread(target=self._update_frame, daemon=True)
        self.thread.start()

    def _update_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    # store a single reference; callers should copy if needed
                    self.frame = frame
            else:
                # avoid tight loop if camera temporarily fails
                time.sleep(self._read_sleep)

    def get_latest_frame(self, copy=True):
        """
        Return the most recent frame. If copy is True a safe copy is returned.
        May return None if no frame has been captured yet.
        """
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy() if copy else self.frame

    def stop(self):
        """Stop background thread and release capture."""
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        try:
            self.cap.release()
        except Exception:
            pass
