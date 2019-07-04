from tkinter import *

counter = 0

root = Tk()

root.title("Rob der Puzzler")

label = Label(root, text="Model Pfad:")

label.pack()

button = Button(root, text='Stop', width=25, command=root.destroy)
button.pack()
root.mainloop()