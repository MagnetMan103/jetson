from flask import Flask, Response
import cv2
import threading

class VideoStreamer:
    """Lightweight MJPEG streamer for network access"""

    def __init__(self, frame_buffer, host='0.0.0.0', port=5000, quality=85):
        """
        Args:
            frame_buffer: FrameBuffer instance to read from
            host: Network interface to bind to ('0.0.0.0' = all interfaces)
            port: Port to serve stream on
            quality: JPEG quality (1-100, lower = faster/smaller)
        """
        self.frame_buffer = frame_buffer
        self.host = host
        self.port = port
        self.quality = quality

        self.app = Flask(__name__)
        self.app.add_url_rule('/video_feed', 'video_feed', self.video_feed)
        self.app.add_url_rule('/', 'index', self.index)

        self.thread = None
        self.running = False

    def generate_frames(self):
        """Generator that yields JPEG frames"""
        while self.running:
            frame = self.frame_buffer.get_latest_frame(copy=False)
            if frame is None:
                continue

            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def video_feed(self):
        """Route that returns the MJPEG stream"""
        return Response(self.generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')

    def index(self):
        """Simple HTML page to view the stream"""
        return '''
        <html>
            <head>
                <title>Hexapod Camera Feed</title>
                <style>
                    body {
                        margin: 0;
                        background: #000;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                    }
                    img { max-width: 100%; max-height: 100vh; }
                </style>
            </head>
            <body>
                <img src="/video_feed">
            </body>
        </html>
        '''

    def start(self):
        """Start the streaming server in a background thread"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=lambda: self.app.run(host=self.host, port=self.port,
                                       debug=False, threaded=True, use_reloader=False),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        """Stop the streaming server"""
        self.running = False
