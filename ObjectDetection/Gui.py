from tkinter import *
import cv2
from PIL import Image, ImageTk


width, height = 800, 600
cap = cv2.VideoCapture(0)

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

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (17, 17), 0)
    thresh = cv2.threshold(gray,scalethresh.get(), 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=scaleerode.get())
    thresh = cv2.dilate(thresh, None, iterations=scaledilate.get())
    img = Image.fromarray(thresh)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()