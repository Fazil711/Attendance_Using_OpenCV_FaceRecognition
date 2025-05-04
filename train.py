from tkinter import *
from tkinter import ttk
from PIL import ImageTk, Image
from student import Student
import os
import mysql.connector
import cv2
import numpy as np
from tkinter import messagebox

class Train:
	def __init__(self, root):
		self.win = root
		self.canvas = Canvas(self.win, width=1000, height=600)
		self.canvas.pack(expand=YES, fill=BOTH)
		self.img_train = Image.open('photos/bg_photo.png').resize((1000, 600), Image.LANCZOS)
		self.ph_train = ImageTk.PhotoImage(self.img_train)
		self.canvas.create_image(0, 0, anchor = NW, image = self.ph_train)
		width = self.win.winfo_screenwidth()
		height = self.win.winfo_screenheight()
		x = int(width/2 - 1000/2)
		y = int(height/2 - 700/2)
		self.win.geometry("1000x600+" + str(x) + "+" + str(y))
		self.win.resizable(width = False, height = False)
		self.win.title("Train")

		self.button = Button(self.canvas, text = "Train now", font = ("times now roman", 20, "bold"), bg = '#ff0866', width = 20, command = self.train_data)
		self.button.place(x = 320, y = 200)
		self.win.mainloop()

	def train_data(self):
		data_dir = ('data')
		path = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

		faces = []
		ids = []

		for image in path:
			img = Image.open(image).convert('L')
			image_np = np.array(img, 'uint8')
			id_a = int(os.path.split(image)[1].split('.')[1])

			faces.append(image_np)
			ids.append(id_a)
			cv2.imshow("Training", image_np)
			cv2.waitKey(1) == 13

		ids = np.array(ids)

		clf = cv2.face.LBPHFaceRecognizer_create()
		clf.train(faces, ids)
		clf.write("classifier.xml")
		cv2.destroyAllWindows()
		messagebox.showinfo("Result!", "Training dataset completed")

'''
xc = Train()
'''
