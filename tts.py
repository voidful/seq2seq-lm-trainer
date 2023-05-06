import asrp
import nlp2
import IPython.display as ipd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
nlp2.download_file(
    'https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/hubert_base_100_lj/g_00500000',
    './')

tokenizer = AutoTokenizer.from_pretrained("voidful/tts_hubert_m2m100")
model = AutoModelForSeq2SeqLM.from_pretrained("voidful/tts_hubert_m2m100")
model.eval()
cs = asrp.Code2Speech(tts_checkpoint='./g_00500000', vocoder='hifigan')
code = 'v_tok_71v_tok_82v_tok_73v_tok_90v_tok_35v_tok_11v_tok_45v_tok_64v_tok_66v_tok_89v_tok_98v_tok_87v_tok_38v_tok_16v_tok_74v_tok_3v_tok_77v_tok_23v_tok_62'
# inputs = tokenizer(["The quick brown fox jumps over the lazy dog."], return_tensors="pt")
# code = tokenizer.batch_decode(model.generate(**inputs,max_length=1024))[0]
code = [int(i) for i in code.replace("</s>","").replace("<s>","").split("v_tok_")[1:]]
print(code)
# ipd.Audio(data=cs(code), autoplay=False, rate=cs.sample_rate)

