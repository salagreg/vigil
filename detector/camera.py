"""
Wrapper robuste autour de cv2.VideoCapture.

Le but : la boucle de detection ne doit jamais planter si la webcam
est debranchee un instant ou si une lecture rate ponctuellement.
Apres plusieurs echecs consecutifs, on tente une reconnexion complete
(fermeture + reouverture du device) plutot que de laisser le thread mourir.
"""

import time

import cv2


class RobustCamera:
    def __init__(self, index=0, max_consecutive_failures=5, reconnect_delay=2.0):
        self.index = index
        self.max_consecutive_failures = max_consecutive_failures
        self.reconnect_delay = reconnect_delay
        self._cap = None
        self._failures = 0
        self._open()

    def _open(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
        self._cap = cv2.VideoCapture(self.index)
        self._failures = 0

    def read(self):
        """Retourne (ok, frame). Ne leve jamais d'exception."""
        try:
            if self._cap is None or not self._cap.isOpened():
                raise RuntimeError("camera non ouverte")
            ok, frame = self._cap.read()
        except Exception:
            ok, frame = False, None

        if not ok:
            self._failures += 1
            if self._failures >= self.max_consecutive_failures:
                print(f"[CAMERA] {self._failures} echecs consecutifs, tentative de reconnexion...")
                time.sleep(self.reconnect_delay)
                self._open()
            return False, None

        self._failures = 0
        return True, frame

    def release(self):
        try:
            if self._cap is not None:
                self._cap.release()
        except Exception:
            pass
