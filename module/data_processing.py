def get_train_valid_dataset(training_args, tokenizer, model_config):
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("squad")
    train_dataset = dataset['validation']
    valid_dataset = dataset['validation']

    # Define function to process data into model inputs
    def process_data_to_model_inputs(batch):
        # Tokenize questions and contexts
        inputs = tokenizer(batch["question"], batch["context"],
                           return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Tokenize answers and create labels
        answer_texts = [i["text"][0] for i in batch["answers"]]
        encoded_answers = tokenizer(answer_texts,
                                    return_tensors="pt")
        labels = encoded_answers["input_ids"]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
    )

    columns = ["input_ids", "labels", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)
    print("train_dataset", train_dataset[0])
    print("valid_dataset", valid_dataset[0])

    return train_dataset, valid_dataset
