import numpy as np
import cv2
import cv2.aruco as aruco
import math

class MarkerTrackingManager:

    # --- Define Tag
    id_to_find = [72,73,74,75]
    marker_size = 5  # - [cm]

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self,R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6


    # Calculates rotation matrix to euler angles
    # The result is the same as MATLAB except the order
    # of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self,R):
        assert (self.isRotationMatrix(R))

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])


    def getMarkerPoints(self,cameraindex):
        returnPoints = []

        # --- Get the camera calibration path
        calib_path = "CameraCalibration"
        camera_matrix = np.loadtxt(calib_path + '/cameraMatrix.txt', delimiter=',')
        camera_distortion = np.loadtxt(calib_path + '/cameraDistortion.txt', delimiter=',')

        # --- 180 deg rotation matrix around the x axis
        R_flip = np.zeros((3, 3), dtype=np.float32)
        R_flip[0, 0] = 1.0
        R_flip[1, 1] = -1.0
        R_flip[2, 2] = -1.0

        # --- Define the aruco dictionary
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        parameters = aruco.DetectorParameters_create()

        # --- Capture the videocamera (this may also be a video or a picture)
        cap = cv2.VideoCapture(cameraindex)
        # -- Set the camera size as the one it was calibrated with
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1250)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 877)

        # -- Font for the text in the image
        font = cv2.FONT_HERSHEY_PLAIN



        # -- Read the camera frame
        ret, frame = cap.read()

        # -- Convert in gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # -- remember, OpenCV stores color images in Blue, Green, Red

        # -- Find all the aruco markers in the image
        corners, ids, rejected = aruco.detectMarkers(image=gray, dictionary=aruco_dict, parameters=parameters,
                                                     cameraMatrix=camera_matrix, distCoeff=camera_distortion)
        print(ids)

        for p in corners:
            for p1 in p:
                returnPoints.append([int(p1[0][0]),int(p1[0][1])])


        returnPoints = self.fourPoints(returnPoints)


        if ids is not None:
            for id in ids:
                if (self.id_to_find.count(id)==1):

                    # -- ret = [rvec, tvec, ?]
                    # -- array of rotation and position of each marker in camera frame
                    # -- rvec = [[rvec_1], [rvec_2], ...]    attitude of the marker respect to camera frame
                    # -- tvec = [[tvec_1], [tvec_2], ...]    position of the marker in camera frame
                    ret = aruco.estimatePoseSingleMarkers(corners, self.marker_size, camera_matrix, camera_distortion)

                    # -- Unpack the output, get only the first
                    rvec, tvec = ret[0][0, 0, :], ret[1][0, 0, :]

                    # -- Draw the detected marker and put a reference frame over it
                    aruco.drawDetectedMarkers(frame, corners,ids)
                    aruco.drawAxis(frame, camera_matrix, camera_distortion, rvec, tvec, 10)


                cap.release()

        return returnPoints, frame

    def fourPoints(self,midpointlist):
        distanceList = list()
        try:
            for p1 in midpointlist:
                for p2 in midpointlist:
                    if(p1!=p2):
                        distance = self.getPointDistance(p1,p2)
                        distanceList.append([int(distance),p1,p2])

            distanceList.sort()

            pointList = list()

            pointList.append(distanceList[len(distanceList)-1][1])
            pointList.append(distanceList[len(distanceList)-1][2])
            pointList.append(distanceList[len(distanceList)-3][1])
            pointList.append(distanceList[len(distanceList)-3][2])

            outlist = []
            # add wight
            for point in pointList:
                out = [int(self.getPointDistance([0,0],point)), [point]]
                outlist.append(out)

            outlist.sort()

            p1 = outlist.pop(0)[1]
            p4 = outlist.pop(len(outlist) - 1)[1]
            if (outlist[0][1][0] < outlist[1][1][0]):
                p2 = outlist.pop(1)[1]
                p3 = outlist.pop(0)[1]
            else:
                p3 = outlist.pop(1)[1]
                p2 = outlist.pop(0)[1]

            return [p1[0], p2[0], p3[0], p4[0]]
        except Exception:
            pass
    pass





    def getPointDistance(self, p1, p2):
        return math.sqrt(math.pow((p2[1] - p1[1]), 2) + math.pow((p2[0] - p1[0]), 2))








if __name__ == '__main__':
    corners, frame = MarkerTrackingManager().getMarkerPoints(1)
    print(corners)
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
















