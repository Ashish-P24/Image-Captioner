import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def calculate_bleu_scores(references, hypothesis):
    smoothing = SmoothingFunction().method1
    return {
        'BLEU-1': round(sentence_bleu(references, hypothesis, weights=(1, 0, 0, 0), smoothing_function=smoothing), 4),
        'BLEU-2': round(sentence_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing), 4),
        'BLEU-3': round(sentence_bleu(references, hypothesis, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing), 4),
        'BLEU-4': round(sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing), 4)
    }


def lcs(X, Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m):
        for j in range(n):
            if X[i] == Y[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])

    return dp[m][n]


def calculate_rouge_l(references, hypothesis):
    scores = []
    for ref in references:
        lcs_len = lcs(ref, hypothesis)
        if lcs_len == 0:
            scores.append(0)
            continue

        precision = lcs_len / len(hypothesis)
        recall = lcs_len / len(ref)

        if precision + recall == 0:
            scores.append(0)
            continue

        scores.append(2 * precision * recall / (precision + recall))

    return round(np.mean(scores), 4)


def evaluate_caption(generated_caption, reference_captions):
    hypothesis = generated_caption.lower().split()
    references = [ref.lower().split() for ref in reference_captions]

    bleu = calculate_bleu_scores(references, hypothesis)
    rouge = calculate_rouge_l(references, hypothesis)

    return {
        'BLEU-1': bleu['BLEU-1'],
        'BLEU-2': bleu['BLEU-2'],
        'BLEU-3': bleu['BLEU-3'],
        'BLEU-4': bleu['BLEU-4'],
        'ROUGE-L': rouge
    }
