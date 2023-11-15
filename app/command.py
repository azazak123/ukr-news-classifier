import numpy as np


def start(model, text_to_input, labels):
    while True:
        text = text_to_input(input("Input a text:"))

        predict = model.predict(text)

        print([{labels[i]: v} for i, v in enumerate(predict[0])])
        print(f"Class: {labels[np.argmax(predict)]}")
