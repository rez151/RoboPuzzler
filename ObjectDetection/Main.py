import cv2
import ObjectDetection.PiceManager as pm
import ObjectDetection.Visualization as vs
from PIL import Image


class Main:
    def startDetection(self, variant=0, imgid=0, cameraindex=1):
        print("Start")
        if variant == 0:
            extractedPices, img_input = pm.PiceManager().getAllPicesbyFrame(cameraindex)
            self.preentation(extractedPices, img_input)
            pass
        if variant == 1:
            path = "TestImages/{}.jpg".format(imgid)
            extractedPices, img_input = pm.PiceManager().getAllPicesbyPath(path)
            self.preentation(extractedPices, img_input)
            pass


    def preentation(self, extractedPices, img_input):
        #file = open("/Volumes/shared/cordinaten.csv", "w")
        print("Output:")
        i = 0
        for piceImg, midpoint, midpointcm, id, _, rotation, _ in extractedPices:
            id = id+1
            piceImg = cv2.resize(piceImg, (224, 224))
            cv2.imwrite('log_img/Visualization/tmp.png',
                        piceImg)
            vs.Visualization().visualheat('log_img/Visualization/tmp.png', id)
            print("ID: {}".format(i) +
                  " X: {:.2f}mm".format(midpointcm[0]) +
                  " Y: {:.2f}mm".format(midpointcm[1]) +
                  " C: {}".format(id) +
                  " R: {:.2f}°".format(rotation))
            #cv2.imshow(str(i), piceImg)
           # file.write(str(id) + "," + str(midpointcm[0]) + "," + str(midpointcm[1]) + "," + str(round(rotation, 2)) + "\n")
            i += 1
        #file.write("end")
       # file.close()
        #status = open("/Volumes/shared/status.txt", "w")
       # status.close()
        self.loadHeatmaps()

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def loadHeatmaps():
        heat_array = []
        for i in range(1, 7):
            image = cv2.imread("log_img/Visualization/{}.jpg".format(i))
            heat_array.append(image)
        return heat_array


if __name__ == '__main__':
    Main().startDetection(variant=1, imgid=2)
