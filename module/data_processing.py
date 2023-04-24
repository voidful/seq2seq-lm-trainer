def get_train_valid_dataset(training_args, tokenizer, model_config):
    # Load dataset
    from datasets import load_dataset
    dataset = load_dataset("squad")
    train_dataset = dataset['train']
    valid_dataset = dataset['validation']

    # Define function to process data into model inputs
    def process_data_to_model_inputs(batch):
        # Tokenize questions and contexts
        inputs = tokenizer(batch["question"], batch["context"], padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Tokenize answers and create labels
        answer_texts = [i["text"][0] for i in batch["answers"]]
        labels = tokenizer(answer_texts, padding=True, truncation=True, add_special_tokens=False,
                           return_tensors="pt").input_ids
        labels = [[-100 if token_id == tokenizer.pad_token_id else token_id for token_id in seq] for seq in labels]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_train_batch_size
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=training_args.per_device_eval_batch_size
    )

    return train_dataset, valid_dataset
