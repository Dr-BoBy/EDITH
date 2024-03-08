from Live import Live
from FaceDetect import FaceDetect
from ObjDetect import ObjDetect
import time

class Main:
    mode = "live"
    run = False
    def __init__(self):
        print("[DEBUG] : Initialisation du Main")

    def ListenInput(self):
        live = Live(0,True)
        faceDetect = FaceDetect()
        objDetect = ObjDetect()
        while True:
            user_input = input()

            if user_input == "s" and not self.run:
                self.run = True
                self.start()
                time.sleep(1)
                print("[DEBUG] : D�marrage")

            elif user_input == "s" and self.run:
                self.run = False
                live.join()
                time.sleep(1)
                print("[DEBUG] : Arret")

            elif user_input == "f" and self.run:
                self.mode = "faceDetect"
                faceDetect.start()
                time.sleep(1)

            elif user_input == "o" and self.run:
                self.mode = "objDetect"
                objDetect.start()
                time.sleep(1)

            elif user_input == "l" and self.run:
                self.mode = "live"
                time.sleep(1)

            elif user_input == "q" and not self.run:
                break

def main():
    main = Main()
    main.ListenInput()
    

if __name__ == "__main__":
    main()