from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import argparse
import json

class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            (Spice(), "SPICE")
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            #print(method, type(method))
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                    total_scores[m] = sc
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score
        print('**********************')
        for key, value in total_scores.items():
            #print(key, value)
            print("%s: %0.3f" % (key, value))

if __name__ == "__main__":
    ### Usage Exmaple
    #ref = {
    #    '1': ['go down the stairs and stop at the bottom .'],
    #    '2': ['this is a cat.']
    #}
    #gt = {
    #    '1': ['Walk down the steps and stop at the bottom.', 'Go down the stairs and wait at the bottom.', 'Once at the bottom, wait.'],
    #    '2': ['It is a cat.', 'There is a cat over there.', 'cat over there.']
    #}
    ## 注意，这里使用的只是一个sample，cider得分较差也是会有。详细请查看论文。
    #scorer = Scorer(ref, gt)
    #scorer.compute_scores()

    parser = argparse.ArgumentParser(description="Calculate BLEU, ROUGE, CIDEr, and SPICE scores.")
    parser.add_argument("json_path", type=str, help="Path to the JSON file containing ground-truth and generated sentences.")
    args = parser.parse_args()

    with open(args.json_path, "r") as f:
        file_content = json.load(f)

    ref = file_content['ref']
    gt  = file_content['gt']

    scorer = Scorer(ref, gt)
    scorer.compute_scores()
    

