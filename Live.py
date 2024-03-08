import cv2
import threading

class Live(threading.Thread):
    frame = 0
    def __init__(self,indexCam,display):
        super(Live, self).__init__()
        print("[DEBUG] : Initialisation du thread live")
        self.stream = cv2.VideoCapture(indexCam)
        if not self.stream.isOpened():
            print("Erreur: Impossible d'ouvrir la webcam.")
            return
        ret, self.frame = self.stream.read()
        self.display = display   

    def run(self):
        from Main import Main

        while Main.run:
            ret, self.frame = self.stream.read()

            # V�rifier si la lecture du flux vid�o est r�ussie
            if not ret:
                print("Erreur: Impossible de lire le flux video.")
                break

            # Afficher le cadre vidéo en direct si on active l'option
            if(self.display):
                cv2.imshow("Webcam", self.frame)

        # Libérer la webcam et fermer les fen�tres d'affichage
        self.stream.release()
        cv2.destroyAllWindows()
