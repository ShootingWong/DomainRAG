import string
from collections import Counter
from typing import Callable

import numpy as np
import regex
# from rouge import Rouge
from rouge_chinese import Rouge
import jieba
rouge = Rouge()

def normalize_answer(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def em(prediction, ground_truth, normalize_fn):
    # print(f'In EM, normalize_fn(prediction) = {normalize_fn(prediction)} normalize_fn(ground_truth)) = {normalize_fn(ground_truth)} float(normalize_fn(prediction) in normalize_fn(ground_truth)) = {float(normalize_fn(prediction) in normalize_fn(ground_truth))}')
    return float(normalize_fn(ground_truth) in normalize_fn(prediction))

def em_strict(prediction, ground_truth, normalize_fn):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))

def f1_en(prediction, ground_truth, normalize_fn):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def f1_zh(prediction, ground_truth, normalize_fn):
    prediction_tokens = ' '.join(jieba.cut(normalize_fn(prediction)))
    ground_truth_tokens = ' '.join(jieba.cut(normalize_fn(ground_truth)))
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def rouge_wrapper_en(prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0
    
def rouge_wrapper_zh(prediction, ground_truth):
    try:
        # print('In rouge_wrapper_zh jieba.cut(prediction) = ', ' '.join(jieba.cut(prediction)))
        # print('In rouge_wrapper_zh jieba.cut(ground_truth) = ', ' '.join(jieba.cut(ground_truth)))
        result = rouge.get_scores(' '.join(jieba.cut(prediction)), ' '.join(jieba.cut(ground_truth)), avg=True)
        # print('In rouge_wrapper_zh rouge = ', result)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def f1_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
    if isinstance(ground_truths[0], str):
        return max([f1_zh(prediction, gt, normalize_fn) for gt in ground_truths])
    else:
        new_gts = []
        for gt in ground_truths:
            new_gts.append(' '.join(gt))
        return max([f1_zh(prediction, gt, normalize_fn) for gt in new_gts])



# def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
#     try:
#         if isinstance(ground_truths[0], str):
#             return max([em(prediction, gt, normalize_fn) for gt in ground_truths])
#         else:
#             new_gts = []
#             for gt in ground_truths:
#                 new_gts.append(' '.join(gt))
#             # print(f'prediction = {prediction} new_gts = {new_gts} exatc_match = {max([em(prediction, gt, normalize_fn) for gt in new_gts])}')
#             return max([em(prediction, gt, normalize_fn) for gt in new_gts])
#     except:
#         # print('prediction = ', prediction)
#         # print('ground_truths = ', ground_truths)
#         None

# def exact_match_strict_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):
#     try:
#         if isinstance(ground_truths[0], str):
#             return max([em_strict(prediction, gt, normalize_fn) for gt in ground_truths])
#         else:
#             new_gts = []
#             for gt in ground_truths:
#                 new_gts.append(' '.join(gt))
#             return max([em_strict(prediction, gt, normalize_fn) for gt in new_gts])
#     except:
#         # print('prediction = ', prediction)
#         # print('ground_truths = ', ground_truths)
#         None

def exact_match_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):

    if isinstance(ground_truths[0], str):
        return max([em(prediction, gt, normalize_fn) for gt in ground_truths])
    else:
        all_scores = []
        for gts in ground_truths:
            score = np.array([em(prediction, gt, normalize_fn) for gt in gts]).mean()
            all_scores.append(score)
        return max(all_scores)
           
    # except:
    #     # print('prediction = ', prediction)
    #     # print('ground_truths = ', ground_truths)
    #     None

def exact_match_strict_score(prediction, ground_truths, normalize_fn: Callable[[str], str] = lambda x: x):

    if isinstance(ground_truths[0], str):
        return max([em_strict(prediction, gt, normalize_fn) for gt in ground_truths])
    else:
        all_scores = []
        for gts in ground_truths:
            score = np.array([em_strict(prediction, gt, normalize_fn) for gt in gts]).mean()
            all_scores.append(score)
        return max(all_scores)
        
def rouge_score(prediction, ground_truths):
    if isinstance(ground_truths[0], list):
        new_gts = []
        for gt in ground_truths:
            new_gts.append(' '.join(gt))
        ground_truths = new_gts
        
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if (
        len(prediction) == 0 or len(ground_truths) == 0
    ):  # check if empty prediction or if there is no hypothesis with len > 0
        # print('In rouge_score prediction = ', prediction)
        # print('In rouge_score ground_truths = ', ground_truths)
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper_zh(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel
    

class Evaluator:
    name: str
    
    def evaluation(
        self, 
        prediction: str,
        answers: list[str], 
    ) -> float:
        
        raise NotImplementedError(
            "score() is not implemented for {} metric".format(self.name)
        )
        

class ExtractiveEvaluator(Evaluator):
    def __init__(self):
        super(ExtractiveEvaluator, self).__init__()
        
    def evaluation(self, prediction, ground_truths):
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths),
            "exact_match_strict": exact_match_strict_score(prediction, ground_truths),
            "f1": f1_score(prediction, ground_truths),
            "rouge": rouge_score(prediction, ground_truths)
        }
            
        return sample_metrics
    
class FactualEvaluator(Evaluator):
    def __init__(self):
        super(FactualEvaluator, self).__init__()
        
    def evaluation(self, prediction, ground_truths, counter_ground_truths):
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        if isinstance(counter_ground_truths, str):
            ground_truths = [counter_ground_truths]
        sample_metrics = {
            "exact_match": exact_match_score(prediction, ground_truths),
            "exact_match_strict": exact_match_strict_score(prediction, ground_truths),
            "f1": f1_score(prediction, ground_truths),
            "rouge": rouge_score(prediction, ground_truths),
            "counter_exact_match": exact_match_score(prediction, counter_ground_truths),
            "counter_exact_match_strict": exact_match_strict_score(prediction, counter_ground_truths),
            "counter_f1": f1_score(prediction, counter_ground_truths),
            "counter_rouge": rouge_score(prediction, counter_ground_truths)
        }
            
            
        return sample_metrics
    
class ConversationEvaluator(Evaluator):
    def __init__(self):
        super(ConversationEvaluator, self).__init__()
        
        
    def evaluation(self, prediction, ground_truths):
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]
        # print(f'Ground_truths = {ground_truths}\nPrediction = {prediction}')
        sample_metrics = {
            "f1": f1_score(prediction, ground_truths),
            "rouge": rouge_score(prediction, ground_truths)
        }
            
        return sample_metrics