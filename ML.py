import tensorflow as tf
import numpy as np
import os


def init():
    BATCH_SIZE = 128
    EPOCHS = 3

    training_images_file = 'gs://mnist-public/train-images-idx3-ubyte'
    training_labels_file = 'gs://mnist-public/train-labels-idx1-ubyte'
    validation_images_file = 'gs://mnist-public/t10k-images-idx3-ubyte'
    validation_labels_file = 'gs://mnist-public/t10k-labels-idx1-ubyte'

    AUTO = tf.data.experimental.AUTOTUNE

    def read_label(tf_bytestring):
        label = tf.io.decode_raw(tf_bytestring, tf.uint8)
        label = tf.reshape(label, [])
        label = tf.one_hot(label, 10)
        return label

    def read_image(tf_bytestring):
        image = tf.io.decode_raw(tf_bytestring, tf.uint8)
        image = tf.cast(image, tf.float32)/256.0
        image = tf.reshape(image, [28*28])
        return image

    def load_dataset(image_file, label_file):
        imagedataset = tf.data.FixedLengthRecordDataset(image_file, 28*28, header_bytes=16)
        imagedataset = imagedataset.map(read_image, num_parallel_calls=16)
        labelsdataset = tf.data.FixedLengthRecordDataset(label_file, 1, header_bytes=8)
        labelsdataset = labelsdataset.map(read_label, num_parallel_calls=16)
        dataset = tf.data.Dataset.zip((imagedataset, labelsdataset))
        return dataset

    def get_training_dataset(image_file, label_file, batch_size):
        dataset = load_dataset(image_file, label_file)
        dataset = dataset.cache()  # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
        dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
        dataset = dataset.repeat() # Mandatory for Keras for now
        dataset = dataset.batch(batch_size, drop_remainder=True) # drop_remainder is important on TPU, batch size must be fixed
        dataset = dataset.prefetch(AUTO)  # fetch next batches while training on the current one (-1: autotune prefetch buffer size)
        return dataset

    def get_validation_dataset(image_file, label_file):
        dataset = load_dataset(image_file, label_file)
        dataset = dataset.cache() # this small dataset can be entirely cached in RAM, for TPU this is important to get good performance from such a small dataset
        dataset = dataset.batch(10000, drop_remainder=True) # 10000 items in eval dataset, all in one batch
        dataset = dataset.repeat() # Mandatory for Keras for now
        return dataset

    # instantiate the datasets
    training_dataset = get_training_dataset(training_images_file, training_labels_file, BATCH_SIZE)
    validation_dataset = get_validation_dataset(validation_images_file, validation_labels_file)

    # For TPU, we will need a function that returns the dataset
    training_input_fn = lambda: get_training_dataset(training_images_file, training_labels_file, BATCH_SIZE)
    validation_input_fn = lambda: get_validation_dataset(validation_images_file, validation_labels_file)

    model = tf.keras.Sequential(
      [
          tf.keras.layers.Input(shape=(28*28,)),
          tf.keras.layers.Dense(10, activation='softmax')
      ])

    checkpoint_path = "training_1"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # print model layers
    model.summary()

    steps_per_epoch = 60000//BATCH_SIZE  # 60,000 items in this dataset
    print("Steps per epoch: ", steps_per_epoch)

    history = model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
                        validation_data=validation_dataset, validation_steps=1,callbacks=[cp_callback])

    model.save('my_model')
init()
