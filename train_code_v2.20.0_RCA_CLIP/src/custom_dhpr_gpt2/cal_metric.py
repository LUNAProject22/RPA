import json
import argparse
import nltk
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

nltk.download('punkt')

def calculate_bleu(reference, candidate):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference], candidate, smoothing_function=smoothing, weights=(0.25, 0.25, 0.25, 0.25))

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def calculate_cider(reference, candidate):
    cider = Cider()
    score, _ = cider.compute_score({0: [reference]}, {0: [candidate]})
    return score

def calculate_spider(reference, candidate):
    spice = Spice()
    score, _ = spice.compute_score({0: [reference]}, {0: [candidate]})
    return score

def main(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    bleu_scores = []
    rouge_scores = []
    cider_scores = []
    spider_scores = []

    for item in data:
        reference = item['ground-truth']
        candidate = item['generated']

        # BLEU
        bleu_score = calculate_bleu(reference, candidate)
        bleu_scores.append(bleu_score)

        # ROUGE
        rouge1, rouge2, rougeL = calculate_rouge(reference, candidate)
        rouge_scores.append((rouge1, rouge2, rougeL))

        # CIDEr
        cider_score = calculate_cider(reference, candidate)
        cider_scores.append(cider_score)

        # SPICE
        spider_score = calculate_spider(reference, candidate)
        spider_scores.append(spider_score)
    
    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(score[0] for score in rouge_scores) / len(rouge_scores)
    avg_rouge2 = sum(score[1] for score in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(score[2] for score in rouge_scores) / len(rouge_scores)
    avg_cider = sum(cider_scores) / len(cider_scores)
    avg_spider = sum(spider_scores) / len(spider_scores)

    print(f"Average BLEU-4 Score: {avg_bleu}")
    print(f"Average ROUGE-1 Score: {avg_rouge1}")
    print(f"Average ROUGE-2 Score: {avg_rouge2}")
    print(f"Average ROUGE-L Score: {avg_rougeL}")
    print(f"Average CIDEr Score: {avg_cider}")
    print(f"Average SPICE Score: {avg_spider}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEU, ROUGE, CIDEr, and SPICE scores.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing ground-truth and generated sentences.")
    args = parser.parse_args()
    main(args.json_path)
