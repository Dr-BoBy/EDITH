import cv2
import time
from FaceDetect import FaceDetect
from ObjDetect import ObjDetect
import queue

class Live():
    frame_queue = queue.LifoQueue()
    def __init__(self,indexCam,display):
        print("[DEBUG] : Initialisation du live")
        self.stream = cv2.VideoCapture(indexCam)
        if not self.stream.isOpened():
            print("Erreur: Impossible d'ouvrir la webcam.")
            return
        self.stream.set(3, 640)
        self.stream.set(4, 480)
        ret, frame = self.stream.read()
        self.display = display
        self.frame_queue.put(frame)

        self.currentProcess = FaceDetect()
        self.currentProcess.start()

    def Start(self):
        from Main import Main

        while True:
            ret, frame = self.stream.read()
            self.frame_queue.put(frame)

            # Verifier si la lecture du flux video est reussie
            if not ret:
                print("Erreur: Impossible de lire le flux video.")
                break
            # Afficher le cadre vidéo en direct si on active l'option
            if(self.display):
                imgToDisplay = frame
                if(Main.mode == "faceDetect"):
                    imgToDisplay = FaceDetect.frame_queue.get()
                elif(Main.mode == "objDetect"):
                    imgToDisplay = ObjDetect.frame_queue.get()

                cv2.putText(frame, "Mode : "+Main.mode, (30, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)
                cv2.imshow("live", imgToDisplay)
            
            
            #Gestion des modes
            key = cv2.waitKey(1) & 0xFF

            if key == ord('f'):
                if Main.mode != "faceDetect":
                    Main.mode = "faceDetect"
                    self.currentProcess.join()
                    while(not FaceDetect.frame_queue.empty()):
                        tempo=FaceDetect.frame_queue.get()
                    self.currentProcess = FaceDetect()
                    self.currentProcess.start()
                    print("[DEBUG] : Passage en mode faceDetect")
                    time.sleep(1)

            elif key == ord('o'):
                if Main.mode != "objeDetect":
                    Main.mode = "objDetect"
                    self.currentProcess.join()
                    while(not ObjDetect.frame_queue.empty()):
                        tempo=ObjDetect.frame_queue.get()
                    self.currentProcess = ObjDetect()
                    self.currentProcess.start()
                    print("[DEBUG] : Passage en mode objectDetect")
                    time.sleep(1)

            elif key == ord('l'):
                if Main.mode != "live":
                    Main.mode = "live"
                    self.currentProcess.join()
                    print("[DEBUG] : Passage en mode live")
                    time.sleep(1)

            elif key == ord('q'):
                Main.mode = "live"
                try :
                    Main.objDetect.join()
                    Main.objDetect.join()
                finally:
                    break
        # Libérer la webcam et fermer les fenetres d'affichage
        self.stream.release()
        cv2.destroyAllWindows()
