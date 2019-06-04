import cv2
import ObjectDetection.PiceManager as pm


class Main:
    def startDetection(self, path):
        print("Start")
        extractedPices, img_input = pm.PiceManager().getAllPicesbyPath(path)
        # extractedPices, img_input = pm.PiceManager().getAllPicesbyFrame(1)
        cv2.imshow("Input", cv2.resize(img_input,(1015, 734)))
        file = open("output/cordinaten.txt", "w")
        print("Output:")
        i = 0
        for piceImg, midpoint, _, id, _, _, rotation, _ in extractedPices:
            cv2.imshow(str(i), piceImg)
            print("ID: " + str(i) +
                  " X: " + str(round((((midpoint[0] * 2) / 60) * 10), 2)) +
                  " Y: " + str(round((((midpoint[1] * 2) / 60) * 10), 2)) +
                  " C: " + id +
                  " R: " + str(round(rotation, 2)) + "°")
            file.write(id + "," + str(midpoint[0]) + "," + str(midpoint[1]) + "," + str(round(rotation, 2)) + "\n")
            i += 1
        file.close()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Main().startDetection("Images/test2.jpg")
