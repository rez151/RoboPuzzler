from tkinter import *
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import time

import cv2


class GUI:
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


        lbl_thresh  = Label(configframe,text="Threschhold: ")
        lbl_thresh.grid(row=0, column=0)

        threshVar = 0
        scalethresh = Scale( configframe, variable=threshVar, orient=HORIZONTAL, from_=0, to=255)
        scalethresh.grid(row=0, column=1)

        lbl_erode = Label(configframe, text="Erode: ")
        lbl_erode.grid(row=1, column=0)

        erodeVar = 0
        scaleerode = Scale(configframe, variable=erodeVar, orient=HORIZONTAL, from_=0, to=15)
        scaleerode.grid(row=1, column=1)

        lbl_dilate = Label(configframe, text="Dilate: ")
        lbl_dilate.grid(row=2, column=0)

        dilateVar = 0
        scaledilate = Scale(configframe, variable=dilateVar, orient=HORIZONTAL, from_=0, to=15)
        scaledilate.grid(row=2, column=1)

        lbl_model  = Label(configframe, text="Model: ")
        lbl_model.grid(row=3, column=0)

        modelPath = StringVar()
        entrymodel = Entry(configframe)
        entrymodel.grid(row=3, column=1)

        def openFile():
            file = filedialog.askopenfile(initialdir="model", title="Select model",
                                          filetypes=(("Model files", "*.h5"), ("All files", "*.*")))
            modelPath = file.name
            entrymodel.insert(0,modelPath)
            pass

        buttonfileopener = Button(configframe,text="...", command=openFile,fg="black")
        buttonfileopener.grid(row=3, column=2)






        delay = 15
        thresh, width, height = getVideo(threshVar, erodeVar, dilateVar)
        can = Canvas(imageframe, width=width, height=height, bg="gray")
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(thresh))
        can.create_image(0, 0, image=photo, anchor=NW)
        can.pack()

        def update():
            thresh, width, height = getVideo(threshVar, erodeVar, dilateVar)
            photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(thresh))
            can.create_image(0, 0, image=photo, anchor=NW)
            can.update()
            root.after(delay, update)

        update()





        root.mainloop()



def donothing():
    print("placeholder")


def getVideo(threshVar,erodeVar,dilateVar):
    frame = cv2.VideoCapture(0)
    if not frame.isOpened():
        raise ValueError("Unable to open video source", 1)

    width = frame.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = frame.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if frame.isOpened():
        _, image = frame.read()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.GaussianBlur(gray, (17, 17), 0)
        thresh = cv2.threshold(gray, threshVar, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=erodeVar)
        thresh = cv2.dilate(thresh, None, iterations=dilateVar)
        return thresh,width,height




if __name__ == '__main__':
    GUI().mainLoop()
