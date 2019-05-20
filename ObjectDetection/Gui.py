from tkinter import *
import cv2
from PIL import Image, ImageTk
import ObjectDetection.trackMarker as tm
import ObjectDetection.PiceManager as pm
import numpy as np


width, height = 800, 600
cap = cv2.VideoCapture(1)

root = Tk()

frame = Frame(root,width=800,height=600)
frame.pack()

configframe = Frame(frame)
configframe.pack(side=LEFT)

lbl_thresh = Label(configframe, text="Threshold: ")
lbl_thresh.grid(row=0, column=0)

scalethresh = Scale(configframe, from_=0, to=255, orient=HORIZONTAL)
scalethresh.grid(row=0, column=1)

lbl_erode = Label(configframe, text="Erode: ")
lbl_erode.grid(row=1, column=0)

scaleerode = Scale(configframe, from_=0, to=15, orient=HORIZONTAL)
scaleerode.grid(row=1, column=1)

lbl_dilate = Label(configframe, text="Dilate: ")
lbl_dilate.grid(row=2, column=0)

scaledilate = Scale(configframe, from_=0, to=15, orient=HORIZONTAL)
scaledilate.grid(row=2, column=1)


imageframe = Frame(frame)
imageframe.pack(side=RIGHT)

lmain = Label(imageframe)
lmain.pack()

def show_extractThrashFrame(frame):
    try:
        if (tm.trackMarker().getMarker().__sizeof__() > 3):
            image_width = int(2070 / 2)
            image_hight = int(1680 / 2)
            pts1 = np.float32((tm.trackMarker().getMarker()))
            # pts1 = np.sort(pts1,0)
            print(pts1)
            pts2 = np.float32([[0, 0], [image_width, 0], [0, image_hight], [image_width, image_hight]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            frame = cv2.warpPerspective(frame, M, (image_width, image_hight))

    except: Exception

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    # gray = cv2.medianBlur(gray, 17)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4)
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)[1]
    thresh = cv2.threshold(gray, scalethresh.get(), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=scaleerode.get())
    thresh = cv2.dilate(thresh, None, iterations=scaledilate.get())
    return thresh

def show_thresh(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    thresh = cv2.threshold(gray, scalethresh.get(), 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=scaleerode.get())
    thresh = cv2.dilate(thresh, None, iterations=scaledilate.get())
    return thresh

def show_resultImage(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    # gray = cv2.medianBlur(gray, 17)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 4)
    # thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_TRUNC)[1]
    thresh = cv2.threshold(gray,scalethresh.get(), 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh = cv2.erode(thresh, None, iterations=scaleerode.get())
    thresh = cv2.dilate(thresh, None, iterations=scaledilate.get())

    _, cnts,  _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])

    for i, ctr in enumerate(sorted_ctrs):
        if (i == 0):
            pass
        else:
            cv2.drawContours(frame, [ctr], 0, (0, 0, 255), 2)

    # pices , image = pm.PiceManager().getAllPicesbyFrame(thresh,frame)
    return frame


def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)


    # thresh = show_thresh(frame)
    thresh = show_extractThrashFrame(frame)
    # thresh = show_resultImage(frame)


    img = Image.fromarray(thresh)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)
    print("##################")



show_frame()
root.mainloop()