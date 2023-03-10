from model import *
#Uncomment below to load from Remote Git directory 
#mnist = tf.keras.datasets.mnist
#(x_train, y_train),(x_test, y_test) = mnist.load_data()

#Load from local directory 
Path = "C:\\Users\\DVEW89\\Desktop\\Deepak\\git_dev\\adnn-ecu\\hdrnn\\python-tf\\mnist.npz"
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data(Path)

train_images = (train_images / 255.0).astype(np.float32)
test_images =  (test_images / 255.0).astype(np.float32)

train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

NUM_EPOCHS = 30
BATCH_SIZE = 10
epochs = np.arange(1, NUM_EPOCHS + 1, 1)
losses = np.zeros([NUM_EPOCHS])
accuracies = np.zeros([NUM_EPOCHS])

m = Model()

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.batch(BATCH_SIZE)

for i in range(NUM_EPOCHS):
    for x, y in train_ds:
        result = m.train((x, y))

    losses[i] = result['loss']
    accuracies[i] = result['accuracy']
    if (i + 1) % 10 == 0:
        print(f"Finished {i + 1} epochs")
        print(f"Loss : {losses[i]:.3f}")
        print(f"Accuracy : {accuracies[i]:.3f}")


#############################################
# CONVERTING THE MODEL TO TF LITE FORMAT ####
#############################################
SAVED_MODEL_DIR = 'saved_model'

tf.saved_model.save(
    m,
    SAVED_MODEL_DIR,
    signatures = {
        'train':
            m.train.get_concrete_function(),
        'infer':
            m.infer.get_concrete_function(),
        'save':
            m.save.get_concrete_function(),
        'restore':
            m.restore.get_concrete_function(),
    })




