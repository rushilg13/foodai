from keras.models import Sequential
from keras.layers import Convolution2D, Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Activation, Flatten
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
import matplotlib.pyplot as plt


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 4, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

print(classifier.summary())

# Step 2 - Preparing the train/test data and training the model

# Code copied from - https://keras.io/preprocessing/image/
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('test',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            class_mode='categorical') 
history=classifier.fit_generator(
        training_set,
        steps_per_epoch=161, 
        epochs=15,
        validation_data=test_set,
        validation_steps=20)# No of images in test set

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

import numpy as np
from keras.preprocessing import image

test=image.load_img('test/burger/2.jpg',target_size = (64, 64))
test=image.img_to_array(test)
test=np.expand_dims(test, axis=0)

result=classifier.predict(test)
training_set.class_indices
print (result)

# Saving the model
model_json = classifier.to_json()
with open("food_model.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('food_model.h5')
