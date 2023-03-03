
import tensorflow as tf

model = tf.keras.models.load_model('trained_model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# tf.lite.experimental.Analyzer.analyze(model_content = tflite_model)

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)