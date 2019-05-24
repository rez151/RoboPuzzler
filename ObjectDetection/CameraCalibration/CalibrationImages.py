import cv2
import os

#__author__ = "Tiziano Fiorenzani"
#__date__ = "01/06/2018"

class CalibrationImages:

    def save_snaps(self,width=0, height=0, name="snapshot", folder="calibrationimages",cameraindex=1):

        cap = cv2.VideoCapture(cameraindex)
        if width > 0 and height > 0:
            print("Setting the custom Width and Height")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
                # ----------- CREATE THE FOLDER -----------------
                folder = os.path.dirname(folder)
                try:
                    os.stat(folder)
                except:
                    os.mkdir(folder)
        except:
            pass

        nSnap   = 0
        w       = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h       = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        fileName    = "%s/%s_%d_%d_" %(folder, name, w, h)
        while True:
            ret, frame = cap.read()

            cv2.imshow('camera', cv2.resize(frame,(640,480)))

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord(' '):
                print("Saving image ", nSnap)
                cv2.imwrite("%s%d.jpg"%(fileName, nSnap), frame)
                nSnap += 1

        cap.release()
        cv2.destroyAllWindows()




    def getCameraCalibration(self,frame_width,frame_height,camera_index):
        # ---- DEFAULT VALUES ---
        SAVE_FOLDER = "calibrationimages"
        FILE_NAME = "snapshot"

        # ----------- PARSE THE INPUTS -----------------



        self.save_snaps(frame_width,frame_height, FILE_NAME, SAVE_FOLDER, camera_index)

        print("Files saved")
        pass


