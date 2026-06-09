import cv2
import threading

class VideoStream:

    def __init__(self, source=0):
        self.stream = cv2.VideoCapture(source)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._update, daemon=True).start()
        return self

    def _update(self):
        while not self.stopped:
            grabbed, frame = self.stream.read()
            with self.lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.lock:
            grabbed = self.grabbed
            frame = self.frame.copy() if self.frame is not None else None
        return grabbed, frame

    def stop(self):
        self.stopped = True
        self.stream.release()