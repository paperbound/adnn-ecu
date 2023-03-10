import tensorflow as tf
import numpy as np

print("TensorFlow version: ", tf.__version__)

IMG_SIZE = 28

class Model(tf.Module):
    
    def __init__(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE), name='flatten'),
            tf.keras.layers.Dense(30, activation='sigmoid', name='dense_1', 
                kernel_initializer='random_normal', bias_initializer='random_normal'),
            tf.keras.layers.Dense(10, activation='sigmoid', name='dense_2',
                kernel_initializer='random_normal', bias_initializer='random_normal')
        ])
    
        self.model.compile(
            optimizer = tf.keras.optimizers.SGD(learning_rate=3.),
            loss = tf.keras.losses.MeanSquaredError(),
            metrics = ['accuracy']
        )

    @tf.function(input_signature = [
        [tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
        tf.TensorSpec([None, 10], tf.float32)],
    ])
    def train(self, data):
        # with tf.GradientTape() as tape:
        #     predict = self.model(x)
        #     model_loss = self.model.loss(y, predict)
        # gradients = tape.gradient(model_loss, self.model.trainable_variables)
        # self.model.optimizer.apply_gradients(
        #     zip(gradients, self.model.trainable_variables))
        result = self.model.train_step(data)
        return result
    
    @tf.function(input_signature = [
        tf.TensorSpec([None, IMG_SIZE, IMG_SIZE], tf.float32),
    ])
    def infer(self, x):
        logits = self.model(x)
        probabilities = tf.nn.softmax(logits, axis = -1)
        return {
            "output": probabilities,
            "logits": logits
        }
    
    @tf.function(input_signature = [
        tf.TensorSpec(shape = [], dtype = tf.string)
    ])
    def save(self, checkpoint_path):
        tensor_names = [weight.name for weight in self.model.weights]
        tensor_to_save = [weight.read_value() for weight in self.model.weights]
        tf.raw_ops.Save(
            filename = checkpoint_path, tensor_names = tensor_names,
            data = tensor_to_save, name = 'save'
        )
        return {
            "checkpoint_path" : checkpoint_path 
        }
    
    @tf.function(input_signature = [
        tf.TensorSpec(shape = [], dtype = tf.string)
    ])
    def restore(self, checkpoint_path):
        restored_tensors = {}
        for var in self.model.weights:
            restored = tf.raw_ops.Restore(
                file_pattern = checkpoint_path, tensor_name = var.name, 
                dt = var.dtype, name = 'restore'
            )
            var.assign(restored)
            restored_tensors[var.name] = restored
        return restored_tensors
