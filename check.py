from datasets import load_dataset
import json
dataset = load_dataset("voidful/NMSQA-CODE")
# train_dataset = dataset['train']
dev_dataset = dataset['dev']
print(dev_dataset[0])
# for i in range(len(dev_dataset)):
# answer_text = dev_dataset[2]['answers']['text'][0]
# try:
#     answer_unit = json.loads(dev_dataset[2]['hubert_100_answer_unit'])[0]['code']
# except:
#     answer_unit = ""
# # gt = 'v_tok_71v_tok_36v_tok_49v_tok_7v_tok_87v_tok_38v_tok_44v_tok_85v_tok_73v_tok_74v_tok_78v_tok_35v_tok_0v_tok_30v_tok_65v_tok_74v_tok_63v_tok_89v_tok_59v_tok_33v_tok_17v_tok_19v_tok_73v_tok_3v_tok_48v_tok_46v_tok_44v_tok_80v_tok_81v_tok_42v_tok_81v_tok_83v_tok_20v_tok_63'
# # gt = [int(i) for i in gt.replace("</s>","").replace("<s>","").split("v_tok_")[1:]]
# # if answer_unit != "":
# print(answer_unit)
    # if answer_unit == gt:
    #     print(answer_text)
    # answer_
# for i in range(len(train_dataset)):
#     try:
#         context = json.loads(train_dataset[i]['hubert_100_context_unit'])['merged_code']
#         answer = json.loads(train_dataset[i]['hubert_100_answer_unit'])['merged_code']
#         question = json.loads(train_dataset[i]['hubert_100_question_unit'])['merged_code']
#     except:
#         question, context, answer = "", "", ""

#     if question == "":
#         if context == "":
#             if answer == "":
#                 continue
#             else:
#                 print(f"question and context error at i={i}")
#         else:
#             if answer == "":
#                 print(f"question and answer error at i={i}")
#             else:
#                 print(f"question has error at i={i}")

#     else:
#         if context == "":
#             if answer == "":
#                 print(f"context and answer error at i={i}")
#             else:
#                 print(f"context error at i={i}")
#         else:
#             if answer == "":
#                 print(f"answer error at i={i}")
#             else:
#                 continue
