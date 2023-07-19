from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer
from transformers import Seq2SeqTrainingArguments

access_token = "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"
tokenizer = AutoTokenizer.from_pretrained(
    "./training_output/Hubert/checkpoint-218990")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "./training_output/Hubert/checkpoint-218990")

model.push_to_hub("long-t5-base-SQA", use_auth_token=access_token)
tokenizer.push_to_hub("long-t5-base-SQA", use_auth_token=access_token)
