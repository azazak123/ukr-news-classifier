import tensorflow as tf
from tensorflow.keras import layers, models


def create(max_words, max_words_sentence, embedding_dim, class_number):
    model = models.Sequential(
        [
            layers.InputLayer(input_shape=(max_words_sentence,)),
            layers.Embedding(max_words, embedding_dim),
            layers.GlobalAveragePooling1D(),
            layers.Dense(8, activation="relu"),
            layers.Dense(class_number, activation="softmax"),
        ]
    )

    model.summary()

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    return model


def train(model, ds, epochs):
    return model.fit(
        ds,
        epochs=epochs,
    )


def evaluate(model, ds):
    loss, accuracy = model.evaluate(ds)

    print("Loss: ", loss)
    print("Accuracy: ", accuracy)
