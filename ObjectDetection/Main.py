import cv2
import ObjectDetection.PiceManager as pm


class Main:
    @staticmethod
    def startDetection(path):
        print("Start")
        # extractedPices, img_input = pm.PiceManager().getAllPicesbyPath(path)
        extractedPices, img_input = pm.PiceManager().getAllPicesbyFrame(1)
        cv2.imshow("Input", cv2.resize(img_input,(1015, 734)))
        file = open("output/cordinaten.txt", "w")
        print("Output:")
        i = 0
        for piceImg, midpoint, midpointcm, _, id, _, _, rotation, _ in extractedPices:
            cv2.imshow(str(i), piceImg)
            print("ID: " + str(i) +
                  " X: " + str(midpointcm[0]) +
                  " Y: " + str(midpointcm[1]) +
                  " C: " + id +
                  " R: " + str(round(rotation, 2)) + "°")
            file.write(id + "," + str(midpointcm[0]) + "," + str(midpointcm[1]) + "," + str(round(rotation, 2)) + "\n")
            i += 1
        file.close()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Main.startDetection("TestImages/testwithallpicesnorotated.jpg")
