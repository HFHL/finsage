#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
from datetime import datetime

def parse_rewrite_file(filepath):
    """
    Parse the experiment file at `filepath` to extract:
      - question id
      - recall
      - at
    Returns a list of tuples: [(question_id, recall, at), ...]
    (Files may contain multiple questions.)
    """
    data = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        pattern = re.compile(
            r"\*+\s+Question\s+(\d+)\s+\*+"           # *** Question ID ***
            r".*?---Question---\s*(.*?)\s*---Rewritten Question---"  
            r".*?---Recall---\s*(\d+)"                 # recall value
            r".*?---At---\s*(\d+)",                    # at value
            re.DOTALL
        )

        matches = pattern.findall(content)
        for match in matches:
            question_id = int(match[0])
            question_text = match[1].strip()  # remove extra whitespace/newlines
            recall = int(match[2])
            at_val = int(match[3])
            data.append((question_id, recall, at_val))
            if recall == 0:
                print(f"Recall=0: Question {question_id}: {question_text}")

    return data

def parse_e2e_file(filepath):
    """
    Parse the experiment file at `filepath` to extract:
      - question id
      - recall
      - at
    Returns a list of tuples: [(question_id, recall, at), ...]
    (Files may contain multiple questions.)
    """
    data = []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        pattern = re.compile(
            r"\*+\s+Question\s+(\d+)\s+\*+"           # *** Question ID ***
            r".*?---Question---\s*(.*?)\s*---Rewritten Question---"  
            r".*?---Answer---\s*(.*?)\n"                # answer
            r".*?---Expected Answer---\s*(.*?)\n"        # expected answer
            r".*?---Score---\s*(0?\.\d+|1(\.0+)?)",
            re.DOTALL
        )

        matches = pattern.findall(content)
        for match in matches:
            question_id = int(match[0])
            question_text = match[1].strip()  # remove extra whitespace/newlines
            answer = match[2].strip()
            expected_answer = match[3].strip()
            score = float(match[4])
            data.append((question_id, question_text, answer, expected_answer, score))

    return data

def parse_rewrite_directory(directory_path):
    """
    Parse all experiment files in the given directory, return
    a list of tuples (question_num, recall, at).
    """
    all_data = []
    # You can adjust the file pattern if needed, e.g. endswith(".txt")
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            file_data = parse_rewrite_file(filepath)
            all_data.extend(file_data)
    return all_data

def parse_e2e_directory(directory_path):
    """
    Parse all experiment files in the given directory, return
    a list of tuples (question_num, recall, at).
    """
    all_data = []
    # You can adjust the file pattern if needed, e.g. endswith(".txt")
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory_path, filename)
            file_data = parse_e2e_file(filepath)
            all_data.extend(file_data)
    return all_data

def compute_stats_rewrite(data):
    """
    Given a list of (question_num, recall, at), compute:
      1) total_questions
      2) questions_with_recall_zero
      3) recall_at_ratio = sum_of_recall / sum_of_at  (if sum_of_at > 0)
    """
    total_questions = len(data)
    questions_with_recall_zero = sum(1 for (_, r, _) in data if r == 0)

    sum_recall = sum(r for (_, r, _) in data)
    sum_at = sum(a for (_, _, a) in data)

    recall_at_ratio = sum_recall / sum_at if sum_at != 0 else 0
    return total_questions, questions_with_recall_zero, recall_at_ratio

def compute_stats_e2e(data):
    """
    Given a list of (question_id, question, answer, expected_answer, score), compute:
      1) total_questions
      2) average_score
    """
    total_questions = len(data)
    sum_score = sum(s for (_, _, _, _, s) in data)
    average_score = sum_score / total_questions
    return total_questions, average_score

def analysis_rewrite():
    dir_path = "test_questions/"
    dir_names = ["eval_rewrite_eval_rewrite_hyde", "eval_rewrite_eval_rewrite_no_hyde"]

    for dir_name in dir_names:
        path = os.path.join(dir_path, dir_name)
        data = parse_rewrite_directory(path)
        stats = compute_stats_rewrite(data)
        print(f"Directory: {dir_name}")
        print(f"  - Total Questions: {stats[0]}")
        print(f"  - Questions with Recall=0: {stats[1]}")
        print(f"  - Recall/At Ratio: {stats[2]}")

def analysis_e2e():
    dir_path = "test_questions/"
    dir_names = ["eval_rewrite_output"]

    for dir_name in dir_names:
        path = os.path.join(dir_path, dir_name)
        data = parse_e2e_directory(path)
        stats = compute_stats_e2e(data)
        print(f"Directory: {dir_name}")
        print(f"  - Total Questions: {stats[0]}")
        print(f"  - Average Score: {stats[1]}")

if __name__ == "__main__":
    analysis_rewrite()
    # analysis_e2e()