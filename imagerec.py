import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

class myCallback(tf.keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.9):
            print("\nReached 90% accuracy so cancelling training!")
            self.model.stop_training = True
callbacks = myCallback()

msint = tf.keras.datasets.fashion_mnist

(trainingimg,traininglabels),(testimg,testlabels) = msint.load_data()
plt.imshow(trainingimg[0])
plt.show()
print(traininglabels[0])

#normalizing
trainingimg = trainingimg / 255.0
testimg = testimg / 255

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),loss="sparse_categorical_crossentropy",metrics=['accuracy'])
model.fit(trainingimg,traininglabels,epochs=15,callbacks=[callbacks])
model.evaluate(testimg,testlabels)
classifications = model.predict(testimg)
# classifications = classifications[4]*255
print(classifications[0])

print(testlabels[0])