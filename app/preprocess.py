import tensorflow as tf


def preprocess(train_data, test_data, max_words, words_in_sentence):
    vectorize_layer = tf.keras.layers.TextVectorization(
        max_tokens=max_words,
        output_mode="int",
        output_sequence_length=words_in_sentence,
    )

    train_text = train_data.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    def text_to_input(text):
        text = tf.expand_dims(tf.convert_to_tensor(text), -1)
        return vectorize_layer(text)

    train_ds = train_data.map(vectorize_text)
    test_ds = test_data.map(vectorize_text)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds, text_to_input
