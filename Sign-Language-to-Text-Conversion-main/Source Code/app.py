from PIL import Image, ImageTk
import tkinter as tk
import cv2
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import operator
import time
import sys
import os
import matplotlib.pyplot as plt

from spellchecker import SpellChecker
from string import ascii_uppercase


class Application:
    def __init__(self):
        dir(SpellChecker)

        spell = SpellChecker()
        self.hs = spell

        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        self.json_file = open("model/model-bw.json", "r")
        self.model_json = self.json_file.read()
        self.json_file.close()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights("model/model-bw.h5")

        self.json_file_dru = open("model/model-bw_dru.json", "r")
        self.model_json_dru = self.json_file_dru.read()
        self.json_file_dru.close()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights("model/model-bw_dru.h5")

        self.json_file_tkdi = open("model/model-bw_tkdi.json", "r")
        self.model_json_tkdi = self.json_file_tkdi.read()
        self.json_file_tkdi.close()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights("model/model-bw_tkdi.h5")

        self.json_file_smn = open("model/model-bw_smn.json", "r")
        self.model_json_smn = self.json_file_smn.read()
        self.json_file_smn.close()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights("model/model-bw_smn.h5")

        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        print("Loaded model from disk")
        self.root = tk.Tk()
        self.root.title("Sign language Detection")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        self.root.geometry("1500x1500")  # 900x1100
        self.panel = tk.Label(self.root)
        self.panel.place(x=50, y=95, width=640, height=310)  # complete camera window
        self.panel2 = tk.Label(self.root)  # initialize image panel
        self.panel2.place(x=460, y=95, width=310, height=310)  # white window

        self.T = tk.Label(self.root)
        self.T.place(x=31, y=17)
        self.T.config(text="Sign Language to Text", font=("courier", 40, "bold"))

        self.photosign = tk.PhotoImage(file='pics/signs.png')
        self.w6 = tk.Label(self.root, image=self.photosign)
        self.w6.place(x=800, y=100)
        self.tx6 = tk.Label(self.root)
        self.tx6.place(x=800, y=17)
        self.tx6.config(text="Sign Languages", font=("Courier", 30, "bold"))

        self.panel3 = tk.Label(self.root)  # Current Symbol
        self.panel3.place(x=500, y=100, width=250, height=250)

        self.panel4 = tk.Label(self.root)  # word
        self.panel4.place(x=500, y=380, width=250, height=60)

        self.panel5 = tk.Label(self.root)  # sentence
        self.panel5.place(x=500, y=460, width=250, height=60)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=500, y=350)
        self.T2.config(text="Detected Sign", font=("Courier", 18, "bold"))

        self.q_btn = tk.Button(self.root, text="QUIT", command=self.destructor, font=("Courier", 25, "bold"))
        self.q_btn.place(x=1300, y=500)

        self.pred_btn = tk.Button(self.root, text="Prediction", command=self.prediction, font=("Courier", 15, "bold"))
        self.pred_btn.place(x=520, y=550)

        self.word_label = tk.Label(self.root, text="Word:", font=("Courier", 18, "bold"))
        self.word_label.place(x=150, y=650)
        self.word_entry = tk.Entry(self.root, font=("Courier", 18))
        self.word_entry.place(x=300, y=650)

        self.sentence_label = tk.Label(self.root, text="Sentence:", font=("Courier", 18, "bold"))
        self.sentence_label.place(x=150, y=700)
        self.sentence_entry = tk.Entry(self.root, font=("Courier", 18))
        self.sentence_entry.place(x=300, y=700)

        self.add_word_btn = tk.Button(self.root, text="Add Word", command=self.add_word, font=("Courier", 15, "bold"))
        self.add_word_btn.place(x=550, y=750)

        self.clear_btn = tk.Button(self.root, text="Clear", command=self.clear, font=("Courier", 15, "bold"))
        self.clear_btn.place(x=450, y=550)

        self.root.bind('<Return>', self.add_word_enter)
        self.root.bind('<Escape>', self.clear)

        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            cv2.rectangle(cv2image, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (0, 255, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            self.current_image2 = imgtk
            self.current_image2 = cv2.resize(self.current_image2, (310, 310))
            cv2.waitKey(1)
            self.panel2.imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.config(image=self.panel2.imgtk)
            self.current_image3 = cv2image
            self.predict(self.current_image3)
        self.root.after(30, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 3))
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 3))
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 3))
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 3))
        prediction = {}
        prediction['blank'] = result[0][0]
        for i in range(26):
            prediction[chr(65 + i)] = result[0][i + 1]
        prediction['dru'] = result_dru[0][0]
        prediction['tkdi'] = result_tkdi[0][0]
        prediction['smn'] = result_smn[0][0]

        self.result = max(prediction.items(), key=operator.itemgetter(1))[0]

        if self.result == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
        self.ct[self.result] += 1
        if self.ct[self.result] > 60:
            if self.result == 'blank':
                if self.blank_flag == 0:
                    print(self.result)
                    self.blank_flag = 1
                    self.sentence_entry.insert(tk.END, " ")
            else:
                print(self.result)
                self.sentence_entry.insert(tk.END, self.result)
                self.blank_flag = 0
        self.current_symbol = self.result
        self.display_symbol(self.current_symbol)

    def display_symbol(self, symbol):
        path = "data/" + symbol + ".PNG"
        image = Image.open(path)
        image = image.resize((250, 250), Image.ANTIALIAS)
        image = ImageTk.PhotoImage(image)
        self.panel3.config(image=image)
        self.panel3.image = image

    def prediction(self):
        print(self.sentence_entry.get())
        sentence = self.sentence_entry.get()
        sentence = sentence.split(' ')
        pred = ""
        for word in sentence:
            pred += self.hs.correction(word)
            pred += " "
        print(pred)
        self.word_entry.delete(0, tk.END)
        self.word_entry.insert(tk.END, pred)

    def destructor(self):
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def clear(self, event=None):
        self.sentence_entry.delete(0, tk.END)

    def add_word(self):
        word = self.word_entry.get().strip().lower()
        if word != "":
            with open("custom_words.txt", "a") as file:
                file.write(word + "\n")
            self.word_entry.delete(0, tk.END)
            print("Word added: ", word)

    def add_word_enter(self, event):
        self.add_word()


if __name__ == '__main__':
    pba = Application()
    pba.root.mainloop()
