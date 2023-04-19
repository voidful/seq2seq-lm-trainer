from nlgeval import NLGEval

nlgeval = NLGEval(no_skipthoughts=True, no_glove=True, metrics_to_omit=["METEOR", "CIDEr"])


def compute_metrics_fn(predictions, labels):
    print("pred_result")
    print("=================================")
    for i in range(10):
        print(labels[i], " ///// ", predictions[i])
    print("=================================")
    return nlgeval.compute_metrics(ref_list=list(map(list, zip(*labels))), hyp_list=predictions)
