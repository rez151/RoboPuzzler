from tkinter import *
from tkinter import messagebox
import ObjectDetection.PiceManager as PiceManager
from PIL import ImageTk, Image
import cv2

frame = Tk()
frame.title("Rob der Puzzler")
frame.geometry('1000x600')
filepath = StringVar()
img = []

def callback():
    pm = PiceManager.PiceManager()
    extractedPices, img_input = pm.getAllPicesbyFrame(1)
    img = Image.open("TestImages/tmp.jpg")
    imgtk = ImageTk.PhotoImage(image=img)
    imgLabel = Label(frame, image=imgtk)
    imgLabel.place(x=10, y=40, width=950, height=550)
    imgLabel.image = imgtk
    frame.update_idletasks()
    if messagebox.askyesno("Rob der Puzzler", "Continue?"):
        file = open("/Volumes/shared/cordinaten.csv", "w")
        print("Output:")
        i = 0
        for piceImg, midpoint, midpointcm, id, _, rotation, _ in extractedPices:
            id = id + 1
            print("ID: {}".format(i) +
                  " X: {:.2f}mm".format(midpointcm[0]) +
                  " Y: {:.2f}mm".format(midpointcm[1]) +
                  " C: {}".format(id) +
                  " R: {:.2f}°".format(rotation))
            file.write(str(id) + "," + str(midpointcm[0]) + "," + str(midpointcm[1]) + "," + str(round(rotation, 2)) + "\n")
            i += 1
        file.write("end")
        file.close()
        status = open("/Volumes/shared/status.txt", "w")
        status.close()
        del pm
    else:
        del pm
        callback()




l2 = Label(frame, text='Camera Index:  ').grid(row=0, column=0)

variable = StringVar(frame)
variable.set(0)
choices = [0, 1]
camIndex = OptionMenu(frame, variable, *choices).grid(row=0, column=1)

btStart = Button(frame, text='Start', command=callback).grid(row=0, column=2)




if __name__ == '__main__':
    frame.mainloop()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

