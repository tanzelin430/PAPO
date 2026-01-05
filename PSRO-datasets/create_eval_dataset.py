"""
从 aops_instruct_train.csv 后40万条中抽取1000条标准答案作为评估集。
标准答案的proof应该得满分1分，用于评估打分模型的准确率。
"""

import pandas as pd
import json
import re
import random
from tqdm import tqdm

# 配置
CSV_PATH = "/home/tanzelin-p/PSRO4math/PSRO-datasets/aops_instruct_train.csv"
OUTPUT_PATH = "/home/tanzelin-p/PSRO4math/PSRO-datasets/sft_scoring_eval.json"
SAMPLE_SIZE = 1000
LAST_N = 400000  # 从后40万条中抽取
SEED = 42

# 评分指令（与训练数据一致）
INSTRUCTION = """## Instruction
Your task is to evaluate the correctness of a mathematical proof. You will be given a problem and a proposed solution.

Please analyze the solution step by step and determine if it is correct. Consider:
1. Is the mathematical reasoning valid?
2. Are all steps properly justified?
3. Does the solution actually prove what was asked?
4. Are there any errors, gaps, or unjustified assumptions?

After your analysis, provide a score in the following format:
- \\boxed{1} if the proof is completely correct
- \\boxed{0.5} if there are minor errors that don't affect the main conclusion
- \\boxed{0} if there are major errors or the proof is incorrect"""

def parse_messages(messages_str):
    """解析messages字符串，提取user问题和assistant回答"""
    try:
        # 尝试单引号格式
        pattern_single = r"\{'content':\s*'((?:[^'\\]|\\.)*)'\s*,\s*'role':\s*'(user|assistant)'\}"
        # 尝试双引号格式
        pattern_double = r'\{\'content\':\s*"((?:[^"\\]|\\.)*)"\s*,\s*\'role\':\s*\'(user|assistant)\'\}'

        matches = re.findall(pattern_single, messages_str, re.DOTALL)
        if len(matches) < 2:
            matches = re.findall(pattern_double, messages_str, re.DOTALL)

        question = None
        proof = None
        for content, role in matches:
            # 恢复转义字符
            content = content.replace('\\n', '\n').replace("\\'", "'").replace('\\"', '"')
            if role == 'user':
                question = content
            elif role == 'assistant':
                proof = content
        return question, proof
    except Exception as e:
        return None, None

def main():
    random.seed(SEED)

    print(f"读取CSV文件...")
    df = pd.read_csv(CSV_PATH)
    total_rows = len(df)
    print(f"总数据量: {total_rows}")

    # 取后40万条
    start_idx = max(0, total_rows - LAST_N)
    df_tail = df.iloc[start_idx:]
    print(f"后{LAST_N}条数据范围: {start_idx} ~ {total_rows}")

    # 先解析所有数据，再从成功解析的数据中随机抽样
    print("解析数据中...")
    parsed_data = []
    for i in tqdm(range(len(df_tail)), desc="解析进度"):
        row = df_tail.iloc[i]
        question, proof = parse_messages(row['messages'])
        if question and proof:
            parsed_data.append((question, proof))

    print(f"成功解析: {len(parsed_data)} 条")

    # 随机抽取
    sample_data = random.sample(parsed_data, min(SAMPLE_SIZE, len(parsed_data)))

    eval_data = []
    for question, proof in sample_data:
        eval_data.append({
            "instruction": INSTRUCTION,
            "input": f"## Problem\n{question}\n\n## Solution\n{proof}",
            "output": "This is a reference solution from the dataset, which should be correct.\n\n\\boxed{1}"
        })

    print(f"抽样: {len(eval_data)} 条")

    # 保存
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(eval_data, f, ensure_ascii=False, indent=2)

    print(f"评估数据已保存到: {OUTPUT_PATH}")
    print(f"评估指标: 打分为1的比例 = 准确率")

if __name__ == "__main__":
    main()
