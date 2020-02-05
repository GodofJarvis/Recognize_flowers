import glob
import random
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class RECOGNIZE_FLOWERS():
    def __init__(self):
        self.num_epochs = 3
        self.batch_size = 40
        self.input_height = 500
        self.input_width = 500

    def image_datagenerator(self, input_filenames):
        input_files = []
        for i in input_filenames:
            input_files.append(self.train_dir + "/" + i)
        counter = 0
        random.shuffle(input_files)
        while True:
            images = np.zeros(
                (self.batch_size, self.input_width, self.input_height, 3))
            labels = []
            if counter+self.batch_size >= len(input_files):
                counter = 0
            for i in range(self.batch_size):
                img = str(input_files[counter + i])
                images[i] = np.array(Image.open(img)) / 255.0
                file_number = img.replace("HE_Challenge_data/data/train/","").replace(".jpg","")
                labels.append(self.train_labels_dict[int(file_number)])
            yield(images, labels)
            counter += self.batch_size


def main():
    recognizeFlowers = RECOGNIZE_FLOWERS()

if __name__== "__main__":
    main()
