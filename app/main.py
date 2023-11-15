from . import load_datasets
from . import preprocess
from . import command
from . import model as md

targetStrToNum = {
    "політика": 0,
    "спорт": 1,
    "новини": 2,
    "бізнес": 3,
    "технології": 4,
}
targetNumToStr = ["політика", "спорт", "інше", "економіка", "технології"]

MAX_WORDS = 25000
MAX_NUMBER_WORDS_IN_SENTENCE = 300
EMBEDDING_DIM = 16
EPOCHS = 5


train_data, test_data = load_datasets.load("FIdo-AI/ua-news", targetStrToNum)

train_ds, test_ds, text_to_input = preprocess.preprocess(
    train_data,
    test_data,
    MAX_WORDS,
    MAX_NUMBER_WORDS_IN_SENTENCE,
)

model = md.create(
    MAX_WORDS, MAX_NUMBER_WORDS_IN_SENTENCE, EMBEDDING_DIM, len(targetNumToStr)
)

md.train(model, train_ds, EPOCHS)

md.evaluate(model, test_ds)

command.start(model, text_to_input, targetNumToStr)
