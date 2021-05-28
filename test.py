import tensorflow as tf

new_model = tf.keras.models.load_model('my_model')

new_model.summary()

training_images_file = 'gs://mnist-public/train-images-idx3-ubyte'
print(training_images_file)


