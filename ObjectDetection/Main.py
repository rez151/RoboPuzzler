import cv2
import ObjectDetection.PiceManager as pm


class Main:
    def startDetection(self,path):
        #extractedPices, img_input = pm.PiceManager().getAllPicesbyPath(path)
        extractedPices, img_input = pm.PiceManager().getAllPicesbyFrame(1)
        cv2.imshow("Input", img_input)
        file =open("output/cordinaten.txt","w")
        for imageID, piceImg, midPoint, id, rotation in extractedPices:
            cv2.imshow(str(imageID),piceImg)
            print("ID: " + str(imageID) + " X: " + str(round((((midPoint[0]*2)/60)*10),2)) + " Y: " + str(round((((midPoint[1]*2)/60)*10),2)) + " C: " + id + " R: "+ str(round(rotation,2))+"°")
            file.write(id+","+str(midPoint[0])+","+str(midPoint[1])+","+str(round(rotation,2))+"\n")
        file.close()
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    Main().startDetection("TestImages/testwithallpicesrotated2.jpg")
