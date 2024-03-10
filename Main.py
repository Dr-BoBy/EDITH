from Live import Live

class Main:
    mode = "faceDetect"
    def __init__(self):
        print("[DEBUG] : Initialisation du Main")
        self.live = Live(0,True)
    
    def Start(self):
        self.live.Start()


def main():
    main = Main()
    main.Start()

    

if __name__ == "__main__":
    main()