import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 256.0, x_test / 256.0

y_train = tf.keras.utils.to_categorical(y_train)
y_test  = tf.keras.utils.to_categorical(y_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(30, activation='sigmoid',
    kernel_initializer='random_normal', bias_initializer='random_normal'),
  tf.keras.layers.Dense(10, activation='sigmoid',
    kernel_initializer='random_normal', bias_initializer='random_normal'),
])

opt = tf.keras.optimizers.SGD(learning_rate=3.)

model.compile(optimizer=opt,
              loss='mse',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=30, batch_size=10)
model.evaluate(x_test, y_test)

# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
