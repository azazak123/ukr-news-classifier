from datasets import load_dataset, Dataset
from ua_datasets import NewsClassificationDataset


def load(dataset_name, labels_map):
    dataset = load_dataset(dataset_name)

    train_data = (
        dataset["train"]
        .map(
            lambda item: {
                "text": item["text"],
                "label": labels_map[item["target"]],
            },
            num_proc=4,
        )
        .to_tf_dataset(columns="text", shuffle=True, label_cols="label", batch_size=16)
    )

    test_data = (
        dataset["test"]
        .map(
            lambda item: {
                "text": item["text"],
                "label": labels_map[item["target"]],
            },
            num_proc=4,
        )
        .to_tf_dataset(columns="text", shuffle=True, label_cols="label", batch_size=16)
    )

    return train_data, test_data


def load_al(labels_map):
    train = NewsClassificationDataset(
        download=True, root="data/", split="train", return_tags=True
    )

    test = NewsClassificationDataset(
        download=True, root="data/", split="test", return_tags=True
    )

    data = {
        "inputs": [text for _, text, _, _ in train],
        "labels": [labels_map[target] for _, _, target, _ in train],
    }
    ds = Dataset.from_dict(data)
    train_data = ds.to_tf_dataset(
        columns="inputs", label_cols="labels", batch_size=16, shuffle=True
    )

    data = {
        "inputs": [text for _, text, _, _ in test],
        "labels": [labels_map[target] for _, _, target, _ in test],
    }
    ds = Dataset.from_dict(data)
    test_data = ds.to_tf_dataset(
        columns="inputs", label_cols="labels", batch_size=16, shuffle=True
    )

    return train_data, test_data
