# T5-seq2seq-trainer

This is a simple example of using the T5 model for sequence-to-sequence tasks, leveraging Hugging Face's `Trainer` for efficient model training.   
The repository includes a configurable interface for dataset processing and evaluation metrics, allowing for seamless adaptation to various tasks and datasets.  

## Features

- Utilize the powerful T5 model for various seq2seq tasks
- Easy configuration for custom dataset processing and evaluation metrics
- Integration with Hugging Face's `Trainer` for efficient training and evaluation

## Usage

1. **Dataset processing**: Modify `data_processing.py` to accommodate your own dataset. The script should take care of loading, preprocessing, and tokenizing the data as required by the T5 model.

2. **Evaluation metric**: Customize the evaluation metric by modifying `eval_metric.py`. This script should implement the necessary logic to compute the desired evaluation metric for your task (e.g., BLEU score, ROUGE score, etc.).

3. **Training and evaluation**: Execute `main.py` to start the training and evaluation process. This script will use the custom dataset processing and evaluation metric functions specified in the previous steps, along with the Hugging Face `Trainer`, to efficiently train and evaluate the T5 model on your task.

## Requirements

- Python 3.6 or later
- Hugging Face Transformers library
- PyTorch
- tqdm

To install the required packages, run:

```
pip install -r requirements.txt
```

## Example

An example dataset and evaluation metric (e.g., machine translation with BLEU score) can be provided in the repository to demonstrate the usage and modification of the data processing and evaluation metric scripts.

## License

This project is licensed under the [MIT License](LICENSE).