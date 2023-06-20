import wave, struct
import numpy as np
import datasets
import transformers
import os
import pyarrow.parquet as pq
from scipy.io.wavfile import write, read
from datasets import Audio
import IPython.display as ipd

dataset = datasets.load_dataset("voidful/NMSQA_audio")
train_set = dataset['train']
dev_set = dataset['dev']
train_dir, dev_dir = "./NMSQA-train-wav", "./NMSQA-dev-wav"
os.makedirs(train_dir, exist_ok=True)
os.makedirs(dev_dir, exist_ok=True)
with open("NMSQA-train.csv", "w") as f:
    f.writelines("path,text\n")
with open("NMSQA-dev.csv", "w") as f:
    f.writelines("path,text\n")

for data in train_set:
    try:
        audio_data = data['content_segment_audio_path']['array']
        audio = ipd.Audio(data=audio_data, autoplay=False, rate=24000)
    except:
        continue
    gt_text = data['content_segment_normalized_text']
    ids = data['id']
    file_name = ids + '.wav'
    with open(os.path.join(train_dir, file_name), "wb") as f:
        f.write(audio.data)
    with open("NMSQA-train.csv", "a") as f:
        f.writelines(os.path.join(train_dir, file_name) + ',' + gt_text + "\n")

for data in dev_set:
    try:
        audio_data = data['content_segment_audio_path']['array']
        audio = ipd.Audio(data=audio_data, autoplay=False, rate=24000)
    except:
        continue
    gt_text = data['content_segment_normalized_text']
    ids = data['id']
    file_name = ids + '.wav'
    with open(os.path.join(dev_dir, file_name), "wb") as f:
        f.write(audio.data)
    with open("NMSQA-dev.csv", "a") as f:
        f.writelines(os.path.join(dev_dir, file_name) + ',' + gt_text + "\n")