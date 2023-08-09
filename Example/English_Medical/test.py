import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.nist_score import sentence_nist
import json
from nltk.util import ngrams


def get_metrics(pred, target):
    turns = len(target)
    bleu_2 = 0
    bleu_4 = 0
    meteor = 0
    nist_2 = 0
    nist_4 = 0
    for index in range(turns):
        pred_utt = pred[index]
        target_utt = target[index]
        min_len = min(len(pred_utt), len(target_utt))
        lens = min(min_len, 4)
        if lens == 0:
            continue
        if lens >= 4:
            bleu_4_utt = sentence_bleu([target_utt], pred_utt, weights=(0.25, 0.25, 0.25, 0.25),
                                       smoothing_function=SmoothingFunction().method1)
            nist_4_utt = sentence_nist([target_utt], pred_utt, 4)
        else:
            bleu_4_utt = 0
            nist_4_utt = 0
        if lens >= 2:
            bleu_2_utt = sentence_bleu([target_utt], pred_utt, weights=(0.5, 0.5),
                                       smoothing_function=SmoothingFunction().method1)
            nist_2_utt = sentence_nist([target_utt], pred_utt, 2)
        else:
            bleu_2_utt = 0
            nist_2_utt = 0

        bleu_2 += bleu_2_utt
        bleu_4 += bleu_4_utt

        meteor += meteor_score([set(target_utt.split(' '))], pred_utt.split(' '))
        nist_2 += nist_2_utt
        nist_4 += nist_4_utt

    bleu_2 /= turns
    bleu_4 /= turns
    meteor /= turns
    nist_2 /= turns
    nist_4 /= turns
    return bleu_2, bleu_4, meteor, nist_2, nist_4


if __name__ == '__main__':
    target = [json.loads(i)['output'] for i in open('./test.jsonl',encoding='utf-8').readlines()]
    pred = [json.loads(i)['output'] for i in open('./output.jsonl',encoding='utf-8').readlines()]
    bleu_2, bleu_4, meteor, nist_2, nist_4 = get_metrics(pred,target)
    print(bleu_2, bleu_4, meteor, nist_2, nist_4)