import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3), activation="relu", input_shape=(500,500,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    # tf.keras.layers.Dropout(.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512,activation="relu"),
    # tf.keras.layers.Dense(1024,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
# print(model.summary())
model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.RMSprop(lr=0.001),metrics=["accuracy"])

#add image augumentation
training_datagen = ImageDataGenerator(rescale=1/255,rotation_range=40,width_shift_range=0.2,

height_shift_range=0.2,shear_range=.2,zoom_range=.2,horizontal_flip=True,fill_mode="nearest")
training_generator = training_datagen.flow_from_directory(
    "./Validation Datasets",target_size=(500,500),batch_size=7, class_mode="binary"
)
validator_datagen = ImageDataGenerator(rescale=1/255)
validator_generator = validator_datagen.flow_from_directory(
    "./Training Datasets",target_size=(500,500),batch_size=7, class_mode="binary"
)

history = model.fit(
    training_generator, validation_data = validator_generator,epochs=15,verbose=1
)

for i in ["boy 1.jpg","boy 2.png","boy 3.png","boy 4.png","boy 5.png","boy 6.png","boy far.png",
"girl 1.png","girl 2.png","girl 3.png","girl 4.png","girl 5.png"]:
    img = image.load_img(os.getcwd()+"\\test\\"+i,target_size=(500,500))
    x = image.img_to_array(img)
    x = x/255
    x = np.expand_dims(x,axis=0)
    images = np.vstack([x])
    classes = model.predict(images)
    print(classes[0])
    if classes[0]>0.5:
        print(i+" is a guy")
    else:
        print(i+" is a girl")
