import json
import argparse
import nltk
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
import stanza

nltk.download('punkt')

# Download and set up stanza
stanza.download('en')
nlp = stanza.Pipeline('en', processors='tokenize,pos,lemma,depparse')

def calculate_bleu(reference, candidate):
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference], candidate, smoothing_function=smoothing)#, weights=(0.25, 0.25, 0.25, 0.25))

def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def calculate_cider(reference, candidate):
    cider = Cider()
    score, _ = cider.compute_score({0: [reference]}, {0: [candidate]})
    return score

def calculate_spice(reference, candidate):
    from collections import defaultdict
    import math
    import numpy as np

    def preprocess(text):
        doc = nlp(text)
        tokens = []
        for sentence in doc.sentences:
            for word in sentence.words:
                tokens.append(word.lemma)
        return ' '.join(tokens)

    def cosine_similarity(vec1, vec2):
        dot_product = sum(p*q for p, q in zip(vec1, vec2))
        magnitude = math.sqrt(sum([val**2 for val in vec1])) * math.sqrt(sum([val**2 for val in vec2]))
        if not magnitude:
            return 0.0
        return dot_product / magnitude

    def build_vector(tokens, idf_dict):
        vec = [idf_dict[token] for token in tokens if token in idf_dict]
        return np.mean(vec, axis=0) if vec else np.zeros(len(idf_dict))

    reference_tokens = preprocess(reference).split()
    candidate_tokens = preprocess(candidate).split()
    all_tokens = list(set(reference_tokens + candidate_tokens))

    idf_dict = defaultdict(lambda: 1.0)  # Using a simple IDF dictionary with 1.0 for all tokens for simplicity

    reference_vector = build_vector(reference_tokens, idf_dict)
    candidate_vector = build_vector(candidate_tokens, idf_dict)

    return cosine_similarity(reference_vector, candidate_vector)

def main(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    bleu_scores = []
    rouge_scores = []
    cider_scores = []
    spice_scores = []

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
        spice_score = calculate_spice(reference, candidate)
        spice_scores.append(spice_score)
    
    # Calculate average scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_rouge1 = sum(score[0] for score in rouge_scores) / len(rouge_scores)
    avg_rouge2 = sum(score[1] for score in rouge_scores) / len(rouge_scores)
    avg_rougeL = sum(score[2] for score in rouge_scores) / len(rouge_scores)
    avg_cider = sum(cider_scores) / len(cider_scores)
    avg_spice = sum(spice_scores) / len(spice_scores)

    print(f"Average BLEU-4 Score: {avg_bleu}")
    print(f"Average ROUGE-1 Score: {avg_rouge1}")
    print(f"Average ROUGE-2 Score: {avg_rouge2}")
    print(f"Average ROUGE-L Score: {avg_rougeL}")
    print(f"Average CIDEr Score: {avg_cider}")
    print(f"Average SPICE Score: {avg_spice}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate BLEU, ROUGE, CIDEr, and SPICE scores.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing ground-truth and generated sentences.")
    args = parser.parse_args()
    main(args.json_path)
