import evaluate
import json

TEST_FILE_PATH='data/cnn/test.jsonl'
PREDICT_PATH='pred_rkl_l16.jsonl'

if __name__ == "__main__":
    preds = []
    with open(PREDICT_PATH) as fin:
        for line in fin:
            d = json.loads(line)
            preds.append(d)

    data = []
    with open(TEST_FILE_PATH) as fin:
        for line in fin:
            d = json.loads(line)
            data.append(d)

    metrics = evaluate.load("metrics/rouge.py")
    for d, p in zip(data, preds):
        metrics.add_batch(predictions=[p['predict']], references=[d['output']])
    result = metrics.compute()
    print(result)
