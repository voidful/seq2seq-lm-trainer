def get_train_valid_dataset(training_args, tokenizer, model_config):
    def strip_answer_array(batch):
        batch["answer"] = batch["answer"][0]
        return batch

    # Load dataset
    from datasets import load_dataset
    from datasets import concatenate_datasets

    # for dataset squad_v2
    squad2_dataset = load_dataset("squad_v2")
    # remove unanswerable question rows
    squad2_dataset['train'] = squad2_dataset['train'].filter(lambda example: example["answers"]["text"] != [])
    squad2_dataset['validation'] = squad2_dataset['validation'].filter(lambda example: example["answers"]["text"] != [])
    # preprocess answers.text array to answer string
    squad2_dataset = squad2_dataset.flatten()
    squad2_dataset = squad2_dataset.rename_column("answers.text", "answer")
    squad2_dataset = squad2_dataset.map(strip_answer_array)
    print(squad2_dataset)
    # print()

    # for dataset superGlue
    superGlue_dataset = load_dataset("super_glue", "multirc")
    # rename "paragraph" as "context"
    superGlue_dataset = superGlue_dataset.rename_column("paragraph", "context")
    print(superGlue_dataset)
    # print()

    # for dataset newsqa
    newsqa_dataset = load_dataset("lucadiliello/newsqa")
    newsqa_dataset = newsqa_dataset.flatten()
    # preprocess answers array to answer string
    newsqa_dataset = newsqa_dataset.rename_column("answers", "answer")
    newsqa_dataset = newsqa_dataset.map(strip_answer_array)
    print(newsqa_dataset)
    # print()

    # for dataset drop
    drop_dataset = load_dataset("drop")
    # rename "passage" as "context"
    drop_dataset = drop_dataset.rename_column("passage", "context")
    # preprocess answers_spans.spans array to answer string
    drop_dataset = drop_dataset.flatten()
    drop_dataset = drop_dataset.rename_column("answers_spans.spans", "answer")
    drop_dataset = drop_dataset.map(strip_answer_array)
    print(drop_dataset)
    # print()

    # for dataset narrativeqa
    narrativeqa_dataset = load_dataset("narrativeqa")
    narrativeqa_dataset = narrativeqa_dataset.flatten()
    # rename "question.text" as "question"
    narrativeqa_dataset = narrativeqa_dataset.rename_column("question.text", "question")
    # rename "document.summary.text" as "context"
    narrativeqa_dataset = narrativeqa_dataset.rename_column("document.summary.text", "context")
    # preprocess several answers dict to answer string
    narrativeqa_dataset = narrativeqa_dataset.rename_column("answers", "answer")
    narrativeqa_dataset = narrativeqa_dataset.map(strip_answer_array)
    narrativeqa_dataset = narrativeqa_dataset.flatten()
    narrativeqa_dataset = narrativeqa_dataset.rename_column("answer.text", "answer")
    print(narrativeqa_dataset)
    # print()

    # split all train & valid dataset
    # & remove all colums except "context" or "question" or "answer"
    # datatype of "context" or "question" or "answer" are all string
    squad2_train_dataset = squad2_dataset['train']
    squad2_valid_dataset = squad2_dataset['validation']
    squad2_train_dataset = squad2_train_dataset.remove_columns([
        "id",
        "title",
        'answers.answer_start',
    ])
    squad2_valid_dataset = squad2_valid_dataset.remove_columns([
        "id",
        "title",
        'answers.answer_start',
    ])

    superGlue_train_dataset = superGlue_dataset['train']
    superGlue_valid_dataset = superGlue_dataset['validation']
    superGlue_train_dataset = superGlue_train_dataset.remove_columns([
        'idx',
        'label',
    ])
    superGlue_valid_dataset = superGlue_valid_dataset.remove_columns([
        'idx',
        'label',
    ])

    newsqa_train_dataset = newsqa_dataset['train']
    newsqa_valid_dataset = newsqa_dataset['validation']
    newsqa_train_dataset = newsqa_train_dataset.remove_columns([
        'key',
        'labels',
    ])
    newsqa_valid_dataset = newsqa_valid_dataset.remove_columns([
        'key',
        'labels',
    ])

    drop_train_dataset = drop_dataset['train']
    drop_valid_dataset = drop_dataset['validation']
    drop_train_dataset = drop_train_dataset.remove_columns([
        'section_id',
        'query_id',
        'answers_spans.types'
    ])
    drop_valid_dataset = drop_valid_dataset.remove_columns([
        'section_id',
        'query_id',
        'answers_spans.types'
    ])

    narrativeqa_train_dataset = narrativeqa_dataset['train']
    narrativeqa_valid_dataset = narrativeqa_dataset['validation']
    narrativeqa_train_dataset = narrativeqa_train_dataset.remove_columns([
        'document.id',
        'document.kind',
        'document.url',
        'document.file_size',
        'document.word_count',
        'document.start',
        'document.end',
        'document.text',
        'question.tokens',
        'document.summary.tokens',
        'document.summary.url',
        'document.summary.title',
        'answer.tokens',
    ])
    narrativeqa_valid_dataset = narrativeqa_valid_dataset.remove_columns([
        'document.id',
        'document.kind',
        'document.url',
        'document.file_size',
        'document.word_count',
        'document.start',
        'document.end',
        'document.text',
        'question.tokens',
        'document.summary.tokens',
        'document.summary.url',
        'document.summary.title',
        'answer.tokens',
    ])

    # print(squad2_train_dataset[0])
    # print(superGlue_train_dataset[0])
    # print(newsqa_train_dataset[0])
    # print(drop_train_dataset[0])
    # print(narrativeqa_train_dataset[0])

    # dump all dataset together
    train_dataset = concatenate_datasets(
        [squad2_train_dataset, superGlue_train_dataset, newsqa_train_dataset, drop_train_dataset,
         narrativeqa_train_dataset])
    valid_dataset = concatenate_datasets(
        [squad2_valid_dataset, superGlue_valid_dataset, newsqa_valid_dataset, drop_valid_dataset,
         narrativeqa_valid_dataset])
    train_dataset = train_dataset.shuffle(seed=42)
    valid_dataset = valid_dataset.shuffle(seed=42)

    # print(len(train_dataset))
    # print(len(valid_dataset))

    # Define function to process data into model inputs
    def process_data_to_model_inputs(batch):
        # Tokenize questions and contexts
        # inputs = tokenizer(batch["question"], batch["background"]+"</s> "+batch["situation"],
        #                 #    padding=True, truncation=True,
        #                    return_tensors="pt")
        inputs = tokenizer(batch["question"], batch["context"],
                           #    padding=True, truncation=True,
                           return_tensors="pt")
        # debug msg
        # print(batch["question"])
        # print(batch["context"])
        # print()
        # print(tokenizer.decode(inputs["input_ids"][0]))
        # print()
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Tokenize answers and create labels
        # answer_texts = [i["text"][0] for i in batch["answers"]]
        # print(batch["answers"])
        answer_texts = batch["answer"]
        encoded_answers = tokenizer(answer_texts,
                                    # padding=True, truncation=True,
                                    return_tensors="pt")
        # print()
        # print(tokenizer.decode(encoded_answers["input_ids"].view(-1)))
        # print()
        # labels = encoded_answers["input_ids"]
        labels = encoded_answers["input_ids"][0]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    # Apply the processing function to the datasets
    train_dataset = train_dataset.map(
        process_data_to_model_inputs,
        # batched=True,
    )
    valid_dataset = valid_dataset.map(
        process_data_to_model_inputs,
        # batched=True,
    )

    columns = ["input_ids", "labels", "attention_mask"]
    train_dataset.set_format(type="torch", columns=columns)
    valid_dataset.set_format(type="torch", columns=columns)
    print("train_dataset", train_dataset[0])
    print("valid_dataset", valid_dataset[0])

    return train_dataset, valid_dataset
