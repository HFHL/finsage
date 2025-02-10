import json
import os
import nest_asyncio
import pandas as pd
from llama_index.core import Response
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core.evaluation import EvaluationResult

# 应用 nest_asyncio 以支持异步操作
nest_asyncio.apply()

# 设置 pandas 显示选项
pd.set_option("display.max_colwidth", 0)

# 配置环境
os.environ["OPENAI_API_KEY"] = "sk-HtHqeXLzYohGrfEA46Cd28761bC8419d90FeD2Bf3aD246B4"
BASE_URL = "https://az.gptplus5.com/v1"  # 你的API基础URL
MODEL_NAME = "gpt-4o-mini"  # 你的模型名称

# 初始化 LLM
llm = OpenAI(
    api_base=BASE_URL,
    api_key="sk-HtHqeXLzYohGrfEA46Cd28761bC8419d90FeD2Bf3aD246B4",
    model=MODEL_NAME,
    temperature=0
)

# 初始化评估器
evaluator = FaithfulnessEvaluator(llm=llm)

def load_json_data(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def create_response_object(answer: str, context: str) -> Response:
    """创建Response对象"""
    from llama_index.core.schema import NodeWithScore, TextNode
    
    # 创建包含上下文的Node
    node = TextNode(text=context)
    node_with_score = NodeWithScore(node=node, score=1.0)
    
    # 创建Response对象
    response = Response(response=answer, source_nodes=[node_with_score])
    return response

def evaluate_answers(retrieval_data, correct_data):
    """评估答案的正确性"""
    evaluation_results = []
    passing_count = 0
    total_count = 0
    skipped_questions = {
        'missing_fields': [],
        'not_in_correct_dict': [],
        'evaluation_error': []
    }
    
    print(f"\nInitial data statistics:")
    print(f"Total questions in retrieval data: {len(retrieval_data)}")
    print(f"Total questions in correct data: {len(correct_data)}")
    
    # 创建问题到答案的映射
    correct_dict = {}
    for item in correct_data:
        if 'question' not in item:
            print(f"Warning: Found item in correct_data without question field: {item}")
            continue
        if 'answer' not in item:
            print(f"Warning: Found item in correct_data without answer field: {item['question']}")
            continue
        if 'content_list' not in item:
            print(f"Warning: Found item in correct_data without content_list field: {item['question']}")
            continue
        correct_dict[item['question']] = item

    print(f"\nValid questions in correct_dict: {len(correct_dict)}")
    
    # 评估每个答案
    for item in retrieval_data:
        if 'question' not in item or 'answer' not in item:
            skipped_questions['missing_fields'].append(item.get('question', 'Unknown question'))
            print(f"Skipping due to missing fields: {item}")
            continue
            
        question = item['question']
        if question not in correct_dict:
            skipped_questions['not_in_correct_dict'].append(question)
            print(f"Skipping because question not found in correct_dict: {question}")
            continue
            
        generated_answer = item['answer']
        reference_item = correct_dict[question]
        reference_answer = reference_item['answer']
        
        try:
            # 创建Response对象
            context = ' '.join(reference_item.get('content_list', []))
            response = create_response_object(generated_answer, context)
            
            # 评估答案
            eval_result = evaluator.evaluate_response(response=response)
            
            # 获取评估结果
            passing = eval_result.passing
            feedback = eval_result.feedback
            
            # 更新统计
            total_count += 1
            if passing:
                passing_count += 1
                
            # 存储结果
            evaluation_results.append({
                'question': question,
                'generated_answer': generated_answer,
                'reference_answer': reference_answer,
                'passing': passing,
                'feedback': feedback
            })
            
            # 打印进度和评估详情
            print(f"\nProcessed question {total_count}:")
            print(f"Question: {question}")
            print(f"Generated Answer: {generated_answer}")
            print(f"Reference Answer: {reference_answer}")
            print(f"Passing: {passing}")
            print(f"Feedback: {feedback}")
            print(f"Current passing rate: {(passing_count/total_count)*100:.2f}%")
            
        except Exception as e:
            skipped_questions['evaluation_error'].append(question)
            print(f"Error evaluating question: {question}")
            print(f"Error: {str(e)}")
            continue
    
    # 打印跳过的问题统计
    print("\nSkipped Questions Summary:")
    print(f"Questions skipped due to missing fields: {len(skipped_questions['missing_fields'])}")
    print(f"Questions skipped due to not found in correct_dict: {len(skipped_questions['not_in_correct_dict'])}")
    print(f"Questions skipped due to evaluation error: {len(skipped_questions['evaluation_error'])}")
    
    return evaluation_results, passing_count, total_count, skipped_questions

def main():
    # 文件路径
    # retrieval_json_path = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/scenarios/6/top10/hyde_reranker.json'
    # correct_answer_json_path = '/root/autodl-tmp/RAG_Agent_vllm_tzh/src/test/QAwithChunks/75_testingset_41updated.json'

    retrieval_json_path = '/root/autodl-tmp/RAG_Agent_finance/src/experiment/QA/result_qwen.json'
    correct_answer_json_path = '/root/autodl-tmp/RAG_Agent_finance/src/experiment/QA/questions_correct_answer.json'
    
    # 加载数据
    retrieval_data = load_json_data(retrieval_json_path)
    correct_data = load_json_data(correct_answer_json_path)
    
    print("Starting evaluation...")
    results, passing_count, total_count, skipped_questions = evaluate_answers(retrieval_data, correct_data)
    
    # 打印总体结果
    print("\nEvaluation Complete!")
    print(f"Total questions evaluated: {total_count}")
    print(f"Questions passed: {passing_count}")
    print(f"Overall passing rate: {(passing_count/total_count)*100:.2f}%")
    
    # 保存详细结果到文件
    # output_path = 'llamaindex_evaluation_results.json'
    output_path = '/root/autodl-tmp/RAG_Agent_finance/src/experiment/QA/llamaindex_qwen.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            'summary': {
                'total_questions': total_count,
                'passing_questions': passing_count,
                'passing_rate': (passing_count/total_count)*100,
                'skipped_questions': {
                    'missing_fields': skipped_questions['missing_fields'],
                    'not_in_correct_dict': skipped_questions['not_in_correct_dict'],
                    'evaluation_error': skipped_questions['evaluation_error']
                }
            },
            'detailed_results': results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\nDetailed results saved to {output_path}")
    
    # 打印一些失败的例子
    print("\nExample of failed evaluations:")
    failed_examples = [r for r in results if not r['passing']]
    for i, example in enumerate(failed_examples[:3], 1):
        print(f"\nExample {i}:")
        print(f"Question: {example['question']}")
        print(f"Generated Answer: {example['generated_answer']}")
        print(f"Reference Answer: {example['reference_answer']}")
        print(f"Feedback: {example['feedback']}")

if __name__ == "__main__":
    main()