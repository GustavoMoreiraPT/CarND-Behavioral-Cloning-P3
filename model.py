from sklearn.model_selection import train_test_split

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle

import os
import csv

samples = []
with open('training_simulator_images/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

correction = 0.2

train_samples, validation_samples = train_test_split(samples*6, test_size=0.2)

print("Train samples: " + str(len(train_samples)))
print("Validation samples: " + str(len(validation_samples)))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    print(num_samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'training_simulator_images/IMG/'+batch_sample[0].split('\\')[-1]
                print(name)
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                #repeat process for left image
                left_name = 'training_simulator_images/IMG/'+batch_sample[1].split('\\')[-1]
                left_image = cv2.imread(left_name)
                left_angle = float(batch_sample[3]) + correction
                images.append(left_image)
                angles.append(left_angle)
                #repeat process for right image
                right_name = 'training_simulator_images/IMG/'+batch_sample[2].split('\\')[-1]
                right_image = cv2.imread(right_name)
                right_angle = float(batch_sample[3]) - correction
                images.append(right_image)
                angles.append(right_angle)
                
                #augmented images
                images.append(cv2.flip(center_image, 1))
                images.append(cv2.flip(left_image, 1))
                images.append(cv2.flip(right_image, 1))
                angles.append(center_angle*-1.0)
                angles.append(left_angle*-1.0)
                angles.append(right_angle*-1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
# Set our batch size
batch_size=6

print(len(train_samples))
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
import math

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0, 0))))
#results into an image of shape 156x316x60
model.add(Conv2D(60, (5,5), activation='relu'))

#results into images of shape 152x312x60
model.add(Conv2D(60, (5,5), activation='relu'))
#pooling layers
#by applying a filter of 2x2, scales down the image to 72x156x60
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#results into an image shape of 68x152x60
model.add(Conv2D(60, (5,5), activation='relu'))
#results into an image shape of 34x76x60
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#results into an image shape of 30x72x30
model.add(Conv2D(30, (5,5), activation='relu'))\
#results into and image shape of 15x36x30
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

#results into an image shape of 11x32x30
#model.add(Conv2D(15, (5,5), activation='relu'))\
#results into and image shape of 6x18x15
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))

#6x18x15 = 720 nodes
model.add(Flatten())
#model.add(Dense(500))
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
history_object = model.fit_generator(train_generator, 
            steps_per_epoch=math.ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=math.ceil(len(validation_samples)/batch_size), 
            epochs=5, verbose=1)

model.save('model.h5')


