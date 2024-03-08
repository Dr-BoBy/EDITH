import cv2
import threading

class ObjDetect(threading.Thread) :
    def __init__(self):
        super().__init__()
        print("[DEBUG] : Initialisation du thread ObjDetect")
        self.classNames = ["Arrière Plan", "Avion", "Vélo", "Oiseau", "Bateau", "Bouteille", "Bus", "Voiture", "Chat", "Chaise", "Vache", "Table", "Chien", "Cheval", "Moto", "Personne", "Plante en pot", "Mouton", "Canapé", "Train", "Ecran"]
        self.net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
        
    def run(self):
        from Main import Main
        from Live import Live

        while Main.mode == "objDetect":
            frame = Live.frame

            # Resize de l'image pour avoir une largeur de 400 max
            frame_resized = cv2.resize(frame, (300,300))

            #Conversion en blob
            blob = cv2.dnn.blobFromImage(frame_resized, 0.007843, (300, 300), 127.5)

            # prediction en utilisant le RN
            self.net.setInput(blob)
            detections = self.net.forward()


            #Taille de l'image 
            cols = frame_resized.shape[1]
            rows = frame_resized.shape[0]

            #Itération sur les objets trouvé
            for i in range(detections.shape[2]):
                #Extraction de la probabilité
                confidence = detections[0, 0, i, 2]

                #Filtrage avec un seuil
                if confidence > 0.75:
                    #Extraction de l'ID de classe
                    class_id = int(detections[0, 0, i, 1])

                    #Extraction de la position
                    xLeftBottom = int(detections[0, 0, i, 3] * cols) 
                    yLeftBottom = int(detections[0, 0, i, 4] * rows)
                    xRightTop   = int(detections[0, 0, i, 5] * cols)
                    yRightTop   = int(detections[0, 0, i, 6] * rows)

                    #Calcul facteur de mise à l'échelle
                    heightFactor = frame.shape[0]/300.0  
                    widthFactor = frame.shape[1]/300.0 

                    #Mise à l'échelle par rapport à l'image d'origine
                    xLeftBottom = int(widthFactor * xLeftBottom) 
                    yLeftBottom = int(heightFactor * yLeftBottom)
                    xRightTop   = int(widthFactor * xRightTop)
                    yRightTop   = int(heightFactor * yRightTop)
                    
                    #Ajout d'un rectangle pour localiser 
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom), (xRightTop, yRightTop),(0, 255, 0))

                    #Ajout du nom de classe
                    label = self.classNames[class_id] + ": " + str(confidence)
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    yLeftBottom = max(yLeftBottom, labelSize[1])
                    cv2.rectangle(frame, (xLeftBottom, yLeftBottom - labelSize[1]),(xLeftBottom + labelSize[0], yLeftBottom + baseLine),(255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (xLeftBottom, yLeftBottom),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),2)

            #Affichage
            cv2.imshow("ObjDetect", frame)
        cv2.destroyWindows("ObjDetect")
    
