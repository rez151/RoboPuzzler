import cv2
import ObjectDetection.PiceManager as pm


class Main:
    def startDetection(self):
        extractedPices, img_input = pm.PiceManager().getAllPices()
        cv2.imshow("Input", img_input)
        file =open("output/cordinaten.txt","w")
        for imageID, piceImg, midPoint, classifierID, rotation in extractedPices:
            cv2.imshow(str(imageID),piceImg)
            print("ID: " + str(imageID) + " X: " + str(midPoint[0]) + " Y: " + str(midPoint[1]) + " C: " + classifierID + " R: "+ str(round(rotation,2))+"°")
            file.write("ID"+","+str(midPoint[0])+","+str(midPoint[1])+","+str(round(rotation,2))+"\n")
        file.close()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Main().startDetection()