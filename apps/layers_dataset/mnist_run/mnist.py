import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
model = keras.Sequential(
    [
        #keras.Input(shape=(28,28,1),batch_size=1,dtype=tf.float32),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu",dtype=tf.float32,input_shape=(28,28,1),batch_size=1),
        layers.MaxPooling2D(pool_size=(2, 2),dtype=tf.float32),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu",dtype=tf.float32),
        layers.MaxPooling2D(pool_size=(2, 2),dtype=tf.float32),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(2, activation="softmax",dtype=tf.float32),
    ]
)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()
model.save("./mnist_02.3.h5")