import json
from nltk.translate.bleu_score import corpus_bleu
from rouge_score import rouge_scorer
import numpy as np
from statistics import mode, StatisticsError

# 加载 JSON 数据
retrieval_json_path = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/scenarios/6/top10/hyde_reranker.json'
correct_answer_json_path = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/75_testingset_75updated.json'

# retrieval_json_path = '/root/autodl-tmp/RAG_Agent_finance/src/experiment/QA/result_qwen.json'
# correct_answer_json_path = '/root/autodl-tmp/RAG_Agent_finance/src/experiment/QA/questions_correct_answer.json'

# 加载 JSON 文件
with open(retrieval_json_path, 'r', encoding='utf-8') as file:
    retrieval_data = json.load(file)

with open(correct_answer_json_path, 'r', encoding='utf-8') as file:
    correct_answer_data = json.load(file)

# 创建问题到答案的映射，确保answer字段存在且不为None
retrieval_dict = {}
for item in retrieval_data:
    if 'answer' in item and item['answer'] is not None and isinstance(item['answer'], str):
        retrieval_dict[item['question']] = item['answer']
    else:
        print(f"Warning: Skipping question due to invalid answer field: {item.get('question', 'Unknown question')}")

correct_dict = {}
for item in correct_answer_data:
    if 'answer' in item and item['answer'] is not None and isinstance(item['answer'], str):
        correct_dict[item['question']] = item['answer']
    else:
        print(f"Warning: Skipping question due to invalid answer field in correct answers: {item.get('question', 'Unknown question')}")

# 找到两个文件中共同的问题
common_questions = set(retrieval_dict.keys()) & set(correct_dict.keys())
print(f"Found {len(common_questions)} common questions between the two files")

# 提取匹配的文本对
generated_answers = []
reference_answers = []
for question in common_questions:
    generated_answer = retrieval_dict[question]
    reference_answer = correct_dict[question]
    
    # 确保答案都是有效的字符串
    if isinstance(generated_answer, str) and isinstance(reference_answer, str):
        generated_answers.append(generated_answer)
        reference_answers.append(reference_answer)

print(f"Processing {len(generated_answers)} valid question pairs")

def get_statistics(scores):
    """计算统计指标"""
    if not scores:
        return {
            'mean': 0.0,
            'max': 0.0,
            'min': 0.0,
            'median': 0.0,
            'mode': 0.0
        }
    
    scores_array = np.array(scores)
    try:
        mode_value = mode(scores)
    except StatisticsError:
        mode_value = np.nan  # 如果没有众数，返回NaN
        
    return {
        'mean': np.mean(scores_array),
        'max': np.max(scores_array),
        'min': np.min(scores_array),
        'median': np.median(scores_array),
        'mode': mode_value
    }

# 计算 BLEU 分数
def compute_bleu(generated_answers, reference_answers):
    if not generated_answers or not reference_answers:
        return {'statistics': get_statistics([]), 'scores': []}
        
    bleu_scores = []
    for i in range(len(generated_answers)):
        try:
            # 将参考答案作为单个引用
            reference = [reference_answers[i].split()]  # 注意：这里将参考答案包装在列表中
            candidate = generated_answers[i].split()
            if candidate and len(reference[0]) > 0:
                bleu_score = corpus_bleu([reference], [candidate])
                bleu_scores.append(bleu_score)
        except Exception as e:
            print(f"Warning: Error computing BLEU score for item {i}: {str(e)}")
            continue

    return {
        'statistics': get_statistics(bleu_scores),
        'scores': bleu_scores
    }

# 计算 ROUGE 分数
def compute_rouge(generated_answers, reference_answers):
    if not generated_answers or not reference_answers:
        empty_stats = get_statistics([])
        return {key: empty_stats for key in ['rouge1', 'rouge2', 'rougeL']}
        
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}

    for i in range(len(generated_answers)):
        try:
            score = scorer.score(reference_answers[i], generated_answers[i])
            for key in rouge_scores:
                rouge_scores[key].append(score[key].fmeasure)
        except Exception as e:
            print(f"Warning: Error computing ROUGE score for item {i}: {str(e)}")
            continue

    return {key: get_statistics(scores) for key, scores in rouge_scores.items()}

# 计算并打印 BLEU 分数统计信息
bleu_results = compute_bleu(generated_answers, reference_answers)
print("\nBLEU Score Statistics:")
print(f"Mean: {bleu_results['statistics']['mean']:.4f}")
print(f"Max: {bleu_results['statistics']['max']:.4f}")
print(f"Min: {bleu_results['statistics']['min']:.4f}")
print(f"Median: {bleu_results['statistics']['median']:.4f}")
print(f"Mode: {bleu_results['statistics']['mode']:.4f}")

# 计算并打印 ROUGE 分数统计信息
rouge_results = compute_rouge(generated_answers, reference_answers)
print("\nROUGE Score Statistics:")
for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
    print(f"\n{rouge_type.upper()}:")
    stats = rouge_results[rouge_type]
    print(f"Mean: {stats['mean']:.4f}")
    print(f"Max: {stats['max']:.4f}")
    print(f"Min: {stats['min']:.4f}")
    print(f"Median: {stats['median']:.4f}")
    print(f"Mode: {stats['mode']:.4f}")

# 打印一些示例比较
# print("\nExample comparisons (first 3 pairs):")
# for i in range(min(3, len(generated_answers))):
#     print(f"\nQuestion pair {i+1}:")
#     print(f"Reference: {reference_answers[i]}")
#     print(f"Generated: {generated_answers[i]}")
#     if i < len(bleu_results['scores']):
#         print(f"BLEU score: {bleu_results['scores'][i]:.4f}")