import numpy as np
import cv2
import threading

class FaceDetect(threading.Thread):
    def __init__(self):
        super(FaceDetect,self).__init__()
        print("[DEBUG] : Initialisation du thread faceDetect")
        self.cascade = cv2.CascadeClassifier(cv2.samples.findFile("haarcascade_frontalface_alt.xml"))
        self.nested = cv2.CascadeClassifier(cv2.samples.findFile("haarcascade_eye.xml"))
        

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
        if len(rects) == 0:
            return []
        rects[:,2:] += rects[:,:2]
        return rects

    def draw_rects(img, rects, color):
        for x1, y1, x2, y2 in rects:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def run(self):
        from Main import Main
        from Live import Live

        while Main.mode=="faceDetect":
            img = Live.frame
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            rects = self.detect(gray, self.cascade)
            vis = img.copy()
            self.draw_rects(vis, rects, (0, 255, 0))
            if not self.nested.empty():
                for x1, y1, x2, y2 in rects:
                    roi = gray[y1:y2, x1:x2]
                    vis_roi = vis[y1:y2, x1:x2]
                    subrects = self.detect(roi.copy(), self.nested)
                    self.draw_rects(vis_roi, subrects, (255, 0, 0))

            cv2.imshow("facedetect", vis)
        cv2.destroyWindows("facedetect")


