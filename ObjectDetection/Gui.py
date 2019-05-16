import tkinter
from tkinter import *
from tkinter import filedialog
import threading

import PIL
from PIL import Image
from PIL import ImageTk
import cv2


class GUI:
    # Standard values
    threshVar = 100
    erodeVar = 8
    dilateVar = 2
    cap = cv2.VideoCapture(0)

    def mainLoop(self):
        root = Tk()
        root.title("Rob der Puzzler")

        frame = Frame(root,width=800,height=600)
        frame.pack()

        configframe = Frame(frame)
        configframe.pack(side=LEFT)

        imageframe = Frame(frame)
        imageframe.pack(side=RIGHT)

        menubar = Menu(root)
        root.config(menu=menubar)


        options = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Options", menu=options)
        options.add_command(label="Exit", command=root.quit)

        filemenu = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Model", menu=filemenu)
        filemenu.add_command(label="Open", command=donothing)

        self.configArea(configframe)

        self.modelArea(configframe)

        self.videoPlayer(imageframe)

        root.mainloop()






    def configArea(self,configframe):
        lbl_thresh = Label(configframe, text="Threshold: ")
        lbl_thresh.grid(row=0, column=0)

        self.scalethresh = Scale(configframe, from_=0, to=255, orient=HORIZONTAL)
        self.scalethresh.set(self.threshVar)
        threshVar = self.scalethresh.get()

        self.scalethresh.grid(row=0, column=1)

        lbl_erode = Label(configframe, text="Erode: ")
        lbl_erode.grid(row=1, column=0)

        self.scaleerode = Scale(configframe, from_=0, to=15, orient=HORIZONTAL)
        self.scaleerode.set(self.erodeVar)
        erodeVar = self.scaleerode.get()
        self.scaleerode.grid(row=1, column=1)

        lbl_dilate = Label(configframe, text="Dilate: ")
        lbl_dilate.grid(row=2, column=0)

        self.scaledilate = Scale(configframe, from_=0, to=15, orient=HORIZONTAL)
        self.scaledilate.set(self.dilateVar)
        dilateVar = self.scaledilate.get()
        self.scaledilate.grid(row=2, column=1)

    def modelArea(self,configframe):
        lbl_model = Label(configframe, text="Model: ")
        lbl_model.grid(row=3, column=0)

        modelPath = StringVar()
        entrymodel = Entry(configframe)
        entrymodel.grid(row=3, column=1)

        buttonfileopener = Button(configframe, text="...", fg="black", command=lambda :self.openFile(entrymodel))
        buttonfileopener.grid(row=3, column=2)


    def openFile(self,entrymodel):
        file = filedialog.askopenfile(initialdir="model", title="Select model",
                                      filetypes=(("Model files", "*.h5"), ("All files", "*.*")))
        modelPath = file.name
        entrymodel.insert(0, modelPath)
        pass


    def getVideo(self):
        frame = cv2.VideoCapture(0)
        if not frame.isOpened():
            raise ValueError("Unable to open video source", 1)

        width = frame.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = frame.get(cv2.CAP_PROP_FRAME_HEIGHT)

        if frame.isOpened():
            _, image = frame.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.GaussianBlur(gray, (17, 17), 0)
            thresh = cv2.threshold(gray, self.threshVar, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=self.erodeVar)
            thresh = cv2.dilate(thresh, None, iterations=self.dilateVar)

            return thresh,width,height

    def videoPlayer(self,imageframe):
        lbl = Label(imageframe)
        lbl.pack()
        self.update(imageframe, lbl)

    def update(self,lbl, delay=10):
        _, frame = self.cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.GaussianBlur(gray, (17, 17), 0)
        thresh = cv2.threshold(gray, self.threshVar, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=self.erodeVar)
        thresh = cv2.dilate(thresh, None, iterations=self.dilateVar)
        img = PIL.Image.fromarray(thresh)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl.imgtk =imgtk
        lbl.config(image=imgtk)
        lbl.after(delay, self.update())



def donothing():
    print("placeholder")

if __name__ == '__main__':

    GUI().mainLoop()