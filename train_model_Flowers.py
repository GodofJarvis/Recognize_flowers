import glob
import random
import pandas as pd
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Dense, Activation, Flatten, Dropout, BatchNormalization


class RECOGNIZE_FLOWERS():
    def __init__(self):
        self.num_epochs = 3
        self.batch_size = 40
        self.input_height = 500
        self.input_width = 500

        self.train_dir = "HE_Challenge_data/data/train"
        self.train_files = []
        for i in range(0,200):
            self.train_files.append(str(i) + ".jpg")
        self.labels = pd.read_csv("HE_Challenge_data/data/train.csv")
        self.train_labels = self.labels[:280]
        self.train_labels_dict = {i:j for i,j in zip(self.train_labels["image_id"], self.train_labels["category"])}

        self.validation_files = []
        for i in range(200,280):
            self.validation_files.append(str(i) + ".jpg")
        self.steps_per_epoch = 5

    def buildAndCompileModel(self):
        self.model = Sequential()
        self.model.add(Conv2D(16, (3,3), padding='same',
                         input_shape=(500,500,3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(16,(3,3)))
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(16,(3,3)))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dense(256))
        self.model.add(Activation('relu'))
        self.model.add(Dense(102, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])

    def train_model(self):
        training_generator = self.image_datagenerator(self.train_files)
        validation_generator = self.image_datagenerator(self.validation_files)

        self.model.fit_generator(training_generator,
                            steps_per_epoch=self.steps_per_epoch,
                            validation_data=validation_generator,
                            validation_steps=2,
                            epochs=self.num_epochs)

        self.model.predict_generator(self.validation_generator, verbose=1)

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
