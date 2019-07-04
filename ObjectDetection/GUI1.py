from tkinter import *
import numpy as np
import cv2
import os
from tkinter import filedialog
from tkinter.filedialog import askdirectory, askopenfilename

import ObjectDetection.PiceManager as PiceManager

from PIL import ImageTk, Image


frame = Tk()
frame.title("Rob der Puzzler")
frame.geometry('1000x600')
filepath = StringVar()
img = []
#w = Canvas(frame, width=200, height=100).pack



def callback():
    extractedPices, img_input = PiceManager.PiceManager().getAllPicesbyPath("TestImages/2.jpg")
    #img = ImageTk.PhotoImage(Image.open(img_input))
    #filename = ImageTk.PhotoImage(img_input)
    #img = Image.fromarray(img_input)
    img = Image.open("TestImages/tmp.jpg")
    imgtk = ImageTk.PhotoImage(image=img)
    #canvas = Canvas(frame, height=100, width=100)
    #canvas.image = imgtk  # <--- keep reference of your image
    #canvas.create_image(0, 0, anchor='nw', image=imgtk)
    #canvas.pack()
    imgLabel = Label(frame, image=imgtk)
    imgLabel.place(x=10, y=40, width=950, height=550)
    imgLabel.image = imgtk
    frame.update_idletasks()




def select():
    print("value is %s" % variable.get())


l2 = Label(frame, text='Camera Index:  ').grid(row=0, column=2)

variable = StringVar(frame)
variable.set(0)
choices = [0, 1]
camIndex = OptionMenu(frame, variable, *choices).grid(row=0, column=3)

btStart = Button(frame, text='Start', command=callback).grid(row=0, column=4)



if __name__ == '__main__':
    frame.mainloop()


