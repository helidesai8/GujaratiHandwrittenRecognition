import cv2
import numpy as np
import pandas as pd
import joblib

try:
    import Tkinter as tk
except:
    import tkinter as tk
    from tkinter import *
    from tkinter.ttk import *
    from tkinter import messagebox
    from PIL import Image, ImageTk, ImageGrab
    import tkinter.font as font
    import pandas as pd
    import PIL
    from PIL import Image, ImageOps
    import numpy as np
    import sys
    import os, cv2
    import csv
    from tkinter import filedialog

    myDir = "..\GujOCR\Output"


    class SampleApp(tk.Tk):
        def __init__(self):
            tk.Tk.__init__(self)
            self._frame = None
            self.switch_frame(StartPage)

        def switch_frame(self, frame_class):
            new_frame = frame_class(self)
            if self._frame is not None:
                self._frame.destroy()
            self._frame = new_frame
            self._frame.pack()


    class StartPage(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)

            def prep():
                # Useful function
                def createFileList(myDir, format='.png'):
                    fileList = []
                    print(myDir)
                    for root, dirs, files in os.walk(myDir, topdown=False):
                        for name in files:
                            if name.endswith(format):
                                fullName = os.path.join(root, name)
                                fileList.append(fullName)
                    return fileList

                columnNames = list()
                for i in range(784):
                    pixel = 'p'
                    pixel += str(i)
                    columnNames.append(pixel)
                l = os.listdir("..\GujOCR\Output")
                print(l)

                dic = {val: idx for idx, val in enumerate(l)}
                print(dic)

                train_data = pd.DataFrame(columns=columnNames)
                train_data.to_csv("train.csv", index=False)
                label_count = list()

                print(len(l))

                for i in range(len(l)):
                    mydir = 'OUTPUT/' + l[i]
                    fileList = createFileList(mydir)
                    for file in fileList:
                        img_file = Image.open(file)  # imgfile.show()
                        width, height = img_file.size
                        format = img_file.format
                        mode = img_file.mode

                        label_count.append(dic[l[i]])
                        inverted_image = img_file.convert('RGB')
                        im_invert = ImageOps.invert(inverted_image)
                        size = (28, 28)
                        new_image = img_file.resize(size)

                        img_grey = new_image.convert('L')
                        value = np.asarray(img_grey.getdata(), dtype=np.int).reshape(
                            (img_grey.size[1], img_grey.size[0]))
                        value = value.flatten()
                        with open("train.csv", 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(value)

                read_data = pd.read_csv('train.csv')
                read_data['Label'] = label_count
                print(read_data)
                messagebox.showinfo("completed successfully.", "Training data saved in train.csv")

            master.title("ગુજરાતી Handwritten Character recognition")
            master.resizable(0, 0)
            master.geometry("846x400")
            master.configure(bg='black')
            tk.Label(text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 23), anchor="center",relief="ridge").place_configure(x=2, y=2)
            canvas1 = Canvas(master, width=530, height=210, bg='#5A79A5')
            tk.Label(text="Main Menu", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 15), anchor="center").place_configure(x=145, y=85)
            canvas = Canvas(master, width=100, height=100, bg="white")
            load = Image.open("ka2.png")
            load = load.resize((100, 100), Image.ANTIALIAS)
            ren = ImageTk.PhotoImage(load)
            img = tk.Label(self, image=ren)
            img.image = ren
            l1 = tk.Label(canvas,image=ren,bg="black")
            b1 = tk.Button(canvas1, text="1. Dataset Collection", bg="#22264b", fg="#e6cf8b",font=("Courier", 15),command=lambda: master.switch_frame(Collect1))
            b2 = tk.Button(canvas1, text="2. Preprocessing the dataset", font=('Courier', 15), bg="#22264b", fg="#e6cf8b",command=prep)
            b3 = tk.Button(canvas1, text="3. Train the model and calculate accuracy", font=('Courier', 15), bg="#22264b",fg="#e6cf8b",command=lambda: master.switch_frame(Train))
            b4 = tk.Button(canvas1, text="4. Prediction", font=('Courier', 15), bg="#22264b", fg="#e6cf8b",command=lambda: master.switch_frame(Predict))
            canvas1.place(x=145, y=150)
            canvas.place(x=10,y=80)
            b1.place(x=15, y=20)
            b2.place(x=15, y=65)
            b3.place(x=15, y=110)
            b4.place(x=15, y=155)
            l1.place(x=0,y=0)


    class Collect1(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            tk.Frame.configure(self, bg="gray")
            tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)
            tk.Label(self,text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 23), anchor="center",relief="ridge").place_configure(x=2, y=2)
            tk.Label(self, text="Collection of dataset", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 15), anchor="center").place_configure(x=145, y=85)
            canvas1 = Canvas(master, width=530, height=210, bg='black')
            b1 = tk.Button(canvas1, text="1. Runtime Dataset upgradation with paint", bg="#22264b", fg="#e6cf8b", font=("Courier", 15),command=lambda: master.switch_frame(Collect2))
            b2 = tk.Button(canvas1, text="2. Upload file", font=('Courier', 15), bg="#22264b", fg="#e6cf8b",command=lambda: master.switch_frame(Collect3))
            canvas2 = Canvas(master, width=50, height=50, bg='black')

            load = Image.open("back3.png")
            load = load.resize((40, 40), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)

            img = tk.Label(self, image=render)
            img.image = render
            b3 = tk.Button(canvas2, text="BACK", font=('Courier', 10), image=render, bg="#22264b", fg="#e6cf8b",
                           height=40,
                           width=40, command=lambda: master.switch_frame(StartPage))
            canvas2.place(x=20, y=310)

            canvas1.place(x=145, y=150)
            b1.place(x=15, y=20)
            b2.place(x=15, y=65)
            b3.place(x=4, y=4)
    class Collect2(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            tk.Frame.configure(self, bg="gray")
            tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)
            tk.Label(self,text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 23), anchor="center",relief="ridge").place_configure(x=2, y=2)
            tk.Label(self, text="Collection of dataset", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 15), anchor="center").place_configure(x=145, y=85)

            def screen_capture():
                import pyscreenshot as ImageGrab
                import time
                import os
                os.startfile("C:/ProgramData/Microsoft/Windows/Start Menu/Programs/Accessories/Paint")
                s1 = t1.get()
                os.chdir("C:/Users/helid/PycharmProjects/GujOCR/folder")
                images_folder = "C:/Users/helid/PycharmProjects/GujOCR/folder/" + s1 + "/"
                time.sleep(5)
                for i in range(0, 1):
                    time.sleep(8)
                    im = ImageGrab.grab(bbox=(60, 170, 400, 550))  # x1,y1,x2,y2
                    print("saved......", i)
                    im.save(images_folder + str(i) + '.png')
                    print("clear screen now and redraw now........")
                messagebox.showinfo("Result", "Capturing screen is completed!!")

            canvas1 = Canvas(master, width=530, height=210, bg='black')
            l1 = tk.Label(canvas1, text="Enter Character: ", font=('Courier', 15), bg='black', fg="#e6cf8b")
            t1 = tk.Entry(canvas1, width=20, border=5)
            b1 = tk.Button(canvas1, text="Submit and Collect with Paint", bg="#22264b", fg="#e6cf8b", font=("Courier", 15), command=screen_capture)
            canvas2 = Canvas(master, width=50,height=50,bg='black')
            load = Image.open("back3.png")
            load = load.resize((40, 40), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)

            img = tk.Label(self, image=render)
            img.image = render
            b3 = tk.Button(canvas2, text="BACK", font=('Courier', 10),image=render, bg="#22264b", fg="#e6cf8b",height=40,
                      width=40,command=lambda: master.switch_frame(Collect1))
            canvas2.place(x=20,y=310)

            canvas1.place(x=145, y=150)
            l1.place(x=75, y=5)
            t1.place(x=290, y=5)
            b1.place(x=75,y=45)
            b3.place(x=4,y=4)

    class Collect3(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            tk.Frame.configure(self, bg="gray")
            tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)
            tk.Label(self,text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 23), anchor="center",relief="ridge").place_configure(x=2, y=2)
            tk.Label(self, text="Collection of dataset", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 15), anchor="center").place_configure(x=145, y=85)
            canvas1 = Canvas(master, width=530, height=210, bg='black')
            l3 = tk.Label(canvas1, text="Upload your boxed image:", font=('Courier', 15), bg='black', fg="#e6cf8b")

            def UploadAction(event=None):
                filename = filedialog.askopenfilename()
                print('Selected:', filename)

                def sort_contours(cnts, method="left-to-right"):
                    reverse = False
                    i = 0
                    if method == "right-to-left" or method == "bottom-to-top":
                        reverse = True
                    if method == "top-to-bottom" or method == "bottom-to-top":
                        i = 1
                    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                    (cnts, boundingBoxes) = zip(
                        *sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
                    return (cnts, boundingBoxes)
                def box_extraction(img_for_box_extraction_path, cropped_dir_path):
                    print("Reading image..")
                    img = cv2.imread(img_for_box_extraction_path, 0)
                    (thresh, img_bin) = cv2.threshold(img, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    img_bin = 255 - img_bin
                    print("Storing binary image to Images/Image_bin.jpg..")
                    cv2.imwrite("Images/Image_bin.jpg", img_bin)
                    print("Applying Morphological Operations..")
                    kernel_length = np.array(img).shape[1] // 40
                    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
                    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
                    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
                    cv2.imwrite("Images/verticle_lines.jpg", verticle_lines_img)
                    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
                    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
                    cv2.imwrite("Images/horizontal_lines.jpg", horizontal_lines_img)
                    alpha = 0.5
                    beta = 1.0 - alpha
                    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
                    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
                    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
                    print("Binary image which only contains boxes: Images/img_final_bin.jpg")
                    cv2.imwrite("Images/img_final_bin.jpg", img_final_bin)
                    contours, hierarchy = cv2.findContours(
                        img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")
                    print("Output stored in Output directiory!")
                    idx = 0
                    for c in contours:
                        x, y, w, h = cv2.boundingRect(c)
                        if (w > 22 and h > 22) and w >= h:
                            idx += 1
                            new_img = img[y:y + h, x:x + w]
                            cv2.imwrite("./Output/" + str(idx) + '.png', new_img)
                    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
                    cv2.imwrite("./Temp/img_contour.jpg", img)
                    messagebox.showinfo("Result", "Data segregated from the box image and stored in OUTPUT directory")
                box_extraction(filename, "./Output/")

            b2 = tk.Button(canvas1, text='Open..',bg="#22264b", fg="#e6cf8b", font=("Courier", 12), command=UploadAction)
            canvas2 = Canvas(master, width=50, height=50, bg='black')
            load = Image.open("back3.png")
            load = load.resize((40, 40), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = tk.Label(self, image=render)
            img.image = render
            b3 = tk.Button(canvas2, text="BACK", font=('Courier', 10), image=render, bg="#22264b", fg="#e6cf8b",height=40, width=40, command=lambda: master.switch_frame(StartPage))

            b3.place(x=4, y=4)
            canvas2.place(x=20, y=310)
            canvas1.place(x=145, y=150)
            l3.place(x=75,y=5)
            b2.place(x=290,y=5)
            b3.place(x=4, y=4)


    class Train(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            tk.Frame.configure(self, bg="gray")
            tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)
            tk.Label(self,text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 23), anchor="center",relief="ridge").place_configure(x=2, y=2)
            tk.Label(self, text="Train And Model Data", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 15), anchor="center").place_configure(x=145, y=85)
            canvas1 = Canvas(master, width=530, height=210, bg='black')
            canvas2 = Canvas(master, width=50, height=50, bg='black')
            load = Image.open("back3.png")
            load = load.resize((40, 40), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)

            img = tk.Label(self, image=render)
            img.image = render
            b3 = tk.Button(canvas2, text="BACK", font=('Courier', 10), image=render, bg="#22264b", fg="#e6cf8b",height=40,width=40, command=lambda: master.switch_frame(StartPage))
            b3.place(x=4, y=4)
            canvas2.place(x=20, y=310)
            canvas1.place(x=145, y=150)


    class Predict(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            tk.Frame.configure(self, bg="gray")
            tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)
            tk.Label(self, text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b",
                     width=44, height=2, font=("Courier", 23), anchor="center", relief="ridge").place_configure(
                x=2, y=2)
            tk.Label(self, text="Prediction of your data", fg="#e8edf3", bg="#22264b", width=44, height=2,
                     font=("Courier", 15), anchor="center").place_configure(x=145, y=85)
            canvas1 = Canvas(master, width=530, height=210, bg='black')
            canvas2 = Canvas(master, width=50, height=50, bg='black')
            b1 = tk.Button(canvas1, text="1. Live Prediction using Blackboard", bg="#22264b", fg="#e6cf8b", font=("Courier", 15), command=lambda: master.switch_frame(Predict2))
            b2 = tk.Button(canvas1, text="2. Upload file and Predict", font=('Courier', 15), bg="#22264b", fg="#e6cf8b", command=lambda: master.switch_frame(Predict3))
            load = Image.open("back3.png")
            load = load.resize((40, 40), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)

            img = tk.Label(self, image=render)
            img.image = render
            b3 = tk.Button(canvas2, text="BACK", font=('Courier', 10), image=render, bg="#22264b", fg="#e6cf8b",height=40, width=40, command=lambda: master.switch_frame(StartPage))
            b3.place(x=4, y=4)
            canvas2.place(x=20, y=310)
            canvas1.place(x=145, y=150)
            b1.place(x=15, y=20)
            b2.place(x=15, y=65)
            b3.place(x=4, y=4)

    class Predict2(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            tk.Frame.configure(self, bg="gray")
            tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)
            tk.Label(self,text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 23), anchor="center",relief="ridge").place_configure(x=2, y=2)
            tk.Label(self, text="Live Prediction using Blackboard", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 15), anchor="center").place_configure(x=145, y=85)
            canvas1 = Canvas(master, width=530, height=210, bg='black')
            canvas2 = Canvas(master, width=50, height=50, bg='black')
            load = Image.open("back3.png")
            load = load.resize((40, 40), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)

            def prediction():

                model = joblib.load("model/digit_recognizer")

                img = ImageGrab.grab(bbox=(130, 500, 500, 700))
                img.save("paint.png")

                im = cv2.imread("paint.png")
                load = Image.open("paint.png")
                load = load.resize((280, 280))
                photo = ImageTk.PhotoImage(load)

                # Labels can be text or images
                img = Label(canvas3, image=photo, width=280, height=280)
                img.image = photo
                img.place(x=0, y=0)

                im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                im_gray = cv2.GaussianBlur(im_gray, (15, 15), 0)

                # Threshold the image
                ret, im_th = cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)
                roi = cv2.resize(im_th, (28, 28), interpolation=cv2.INTER_AREA)

                rows, cols = roi.shape

                X = []

                ## Add pixel one by one into data array
                for i in range(rows):
                    for j in range(cols):
                        k = roi[i, j]
                        if k > 100:
                            k = 1
                        else:
                            k = 0
                        X.append(k)

                predictions = model.predict([X])

                a1 = tk.Label(canvas3, text="Prediction= ", font=("verdana", 20))
                a1.place(x=5, y=350)

                b1 = tk.Label(canvas3, text=predictions[0], font=("verdana", 20))
                b1.place(x=200, y=350)

            canvas4 = Canvas(master, width=530, height=210, bg='black')
            canvas4.place(x=145, y=150)

            def activate_paint(e):
                global lastx, lasty
                canvas4.bind('<B1-Motion>', paint)
                lastx, lasty = e.x, e.y

            def paint(e):
                global lastx, lasty
                x, y = e.x, e.y
                canvas4.create_line((lastx, lasty, x, y), width=40, fill="white")
                lastx, lasty = x, y

            canvas4.bind('<1>', activate_paint)

            def clear():
                canvas4.delete("all")

            tk.Label(canvas4, text="Draw in the below blackboard", fg="#e8edf3", bg="#22264b", width=40, height=1,font=("Courier", 15), anchor="center").place_configure(x=50, y=0)
            btn = tk.Button(canvas4, text="clear", fg="white", bg="green", command=clear)
            btn.place(x=0, y=0)

            canvas3 = Canvas(master, width=280, height=530, bg="green")
            # canvas3.place(x=515, y=120)
            img = tk.Label(self, image=render)
            img.image = render

            b3 = tk.Button(canvas2, text="BACK", font=('Courier', 10), image=render, bg="#22264b", fg="#e6cf8b",height=40,width=40, command=lambda: master.switch_frame(Predict))
            b3.place(x=4, y=4)
            canvas2.place(x=20, y=310)
            canvas1.place(x=145, y=150)

    class Predict3(tk.Frame):
        def __init__(self, master):
            tk.Frame.__init__(self, master)
            tk.Frame.configure(self, bg="gray")
            tk.Label(self, text="", font=('Helvetica', 18, "bold")).pack(side="top", padx=750, pady=500)
            tk.Label(self,text="Gujarati Handwritten Character Recognition", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 23), anchor="center",relief="ridge").place_configure(x=2, y=2)
            tk.Label(self, text="Upload file and Predict", fg="#e8edf3", bg="#22264b", width=44, height=2,font=("Courier", 15), anchor="center").place_configure(x=145, y=85)

            canvas1 = Canvas(master, width=530, height=210, bg='black')
            l3 = tk.Label(canvas1, text="Upload your image to be predicted:", font=('Courier', 15), bg='black', fg="#e6cf8b")
            def UploadAction(event=None):
                filename = filedialog.askopenfilename()
                print('Selected:', filename)
            b2 = tk.Button(canvas1, text='Open..', bg="#22264b", fg="#e6cf8b", font=("Courier", 12),command=UploadAction)
            canvas2 = Canvas(master, width=50, height=50, bg='black')
            load = Image.open("back3.png")
            load = load.resize((40, 40), Image.ANTIALIAS)
            render = ImageTk.PhotoImage(load)
            img = tk.Label(self, image=render)
            img.image = render
            b3 = tk.Button(canvas2, text="BACK", font=('Courier', 10), image=render, bg="#22264b", fg="#e6cf8b",height=40, width=40, command=lambda: master.switch_frame(Predict))
            b3.place(x=4, y=4)
            canvas2.place(x=20, y=310)
            canvas1.place(x=145, y=150)
            l3.place(x=75, y=5)
            b2.place(x=220, y=40)
            b3.place(x=4, y=4)

if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()