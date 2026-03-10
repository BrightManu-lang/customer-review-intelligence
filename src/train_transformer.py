import numpy as np
import evaluate
from datasets import load_dataset, DatasetDict
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer)

MODEL_NAME = "distilbert-base-uncased"
DATA_FILE = "data/reviews_binary.csv"
OUTPUT_DIR = "models/distilbert-review-sentiment"


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def compute_metrics(eval_pred):
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=predictions, references=labels)["f1"],
    }


def train_model():
    dataset = load_dataset("csv", data_files=DATA_FILE)["train"]

    split_1 = dataset.train_test_split(test_size=0.2, seed=42)
    split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)

    dataset = DatasetDict({
        "train": split_1["train"],
        "validation": split_2["train"],
        "test": split_2["test"],
    })

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_dataset = dataset.map(
        lambda batch: tokenize_function(batch, tokenizer),
        batched=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "NEGATIVE", 1: "POSITIVE"},
        label2id={"NEGATIVE": 0, "POSITIVE": 1},
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training sentiment model...")
    trainer.train()

    val_metrics = trainer.evaluate()
    test_metrics = trainer.evaluate(tokenized_dataset["test"])

    print("\nValidation metrics:", val_metrics)
    print("\nTest metrics:", test_metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Saved model to: {OUTPUT_DIR}")

    return {"validation": val_metrics, "test": test_metrics}


if __name__ == "__main__":
    train_model()