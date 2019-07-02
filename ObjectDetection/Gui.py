from tkinter import *
import cv2
from PIL import Image, ImageTk
import ObjectDetection.MarkerTrackingManager as tm
import ObjectDetection.PiceManager as pm
import numpy as np
import ObjectDetection.Visualization as vs


cameraIndex = 0

width, height = 800, 600
cap = cv2.VideoCapture(cameraIndex)

root = Tk()

frame = Frame(root, width=800, height=600)
frame.pack()

configframe = Frame(frame)
configframe.pack(side=LEFT)

# imageframe = Frame(frame)
# imageframe.pack(side=RIGHT)

# lmain = Label(imageframe)
# lmain.pack()

@staticmethod
def show_extractThrashFrame(frame):
    try:
        corners = tm.MarkerTrackingManager().getMarkerPoints(0)[0]
        if (len(corners) == 4):
            image_width = int(1080)
            image_hight = int(720)
            pts1 = np.float32(corners)
            pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, M, (image_width, image_hight))
    except Exception as e:
        print(e)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    return thresh

# loading heatmaps into an array of images
def loadHeatmaps():
    heat_array = []
    for i in range(1, 3):
        heat = Image.open("log_img/Visualization/{}.jpg".format(i))
        heat_array.append(heat)
    return heat_array


def getFinalResult(cameraindex=1, imgid=0):
    # extractedPices, img_input = pm.PiceManager().getAllPicesbyFrame(cameraindex)
    path = "TestImages/{}.jpg".format(imgid)
    extractedPices, img_input = pm.PiceManager().getAllPicesbyPath(path)
    i = 0
    for piceImg, midpoint, midpointcm, id, _, rotation, _ in extractedPices:
        i = i + 1
        piceImg = cv2.resize(piceImg, (224, 224))
        cv2.imwrite('log_img/Visualization/tmp.png',
                    piceImg)
        #vs.Visualization().visualheat('log_img/Visualization/tmp.png', id)
        print("ID: {}".format(i) +
              " X: {:.2f}mm".format(midpointcm[0]) +
              " Y: {:.2f}mm".format(midpointcm[1]) +
              " C: {}".format(id) +
              " R: {:.2f}°".format(rotation))


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)

    getFinalResult(imgid=0)
    #heatmap_list = loadHeatmaps()

    print("##################")


show_frame()
image_array = loadHeatmaps()
imgtk = ImageTk.PhotoImage(image=image_array[0])
w = Label(root, image=imgtk).pack()
imgtk = ImageTk.PhotoImage(image=image_array[1])
w = Label(root, image=imgtk).pack()



root.mainloop()
