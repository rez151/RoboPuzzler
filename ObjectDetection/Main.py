import cv2
import ObjectDetection.PiceManager as pm
import ObjectDetection.Visualization as vs


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

    @staticmethod
    def preentation(extractedPices, img_input):
        cv2.imshow("Input", img_input)
        file = open("output/cordinaten.txt", "w")
        print("Output:")
        i = 0
        for piceImg, midpoint, midpointcm, id, _, rotation, _ in extractedPices:
            id = id+1
            #cv2.imshow(str(i), piceImg)
            piceImg= cv2.resize(piceImg, (224, 224))
            cv2.imwrite('log_img/Visualization/tmp.png',
                        piceImg)
            vs.Visualization().visualheat('log_img/Visualization/tmp.png', id)
            print("ID: {}".format(i) +
                  " X: {:.2f}mm".format(midpointcm[0]) +
                  " Y: {:.2f}mm".format(midpointcm[1]) +
                  " C: {}".format(id) +
                  " R: {:.2f}°".format(rotation))

            file.write(str(id) + "," + str(midpointcm[0]) + "," + str(midpointcm[1]) + "," + str(round(rotation, 2)) + "\n")
            i += 1
        file.write("end")
        file.close()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Main().startDetection(variant=1, imgid=2)
