import json
import re
from typing import List, Optional

def extract_answer(text: str) -> str | None:
    """
    提取"答案是"之后的内容（不需要括号）
    适配以下格式：
    - 答案是选项A
    - 答案是：正确答案
    - 答案是 42
    """
    match = re.search(
        r'答案是[\s:：]*(\S+)',  # 匹配冒号/空格后的非空内容
        text
    )
    return match.group(1).strip() if match else None

def extract_uppercase_letters(text: str) -> Optional[List[str]]:
    """
    提取经过extract_answer处理后的内容中的连续大写字母A-F
    （例如从"选项A"提取["A"], 从"ABCD"提取["A","B","C","D"]）
    """
    answer = extract_answer(text)
    if answer is None:
        return None
    
    # 匹配所有连续的A-F大写字母（按单个字母拆分）
    matches = re.findall(r'[ABCDEF]', answer)
    return matches if matches else []


if __name__ == "__main__":
    with open("./distill-EQ-IQ.jsonl", "r", encoding="utf-8") as f:
        data = f.readlines()

    EQ_results = []
    for i in range(80):
        item = json.loads(data[i])
        llm_answer = extract_uppercase_letters(item["output"].split("</think>")[-1])
        if llm_answer:
            llm_answer = llm_answer[0]
        else:
            llm_answer = "None"
        correct_answer = item["correct"]
        EQ_results.append([llm_answer, correct_answer])

    IQ_results = []
    for i in range(80, len(data)):
        item = json.loads(data[i])
        llm_answer = extract_uppercase_letters(item["output"].split("</think>")[-1])
        if llm_answer:
            llm_answer = llm_answer[0]
        else:
            llm_answer = "None"
        correct_answer = item["correct"]
        IQ_results.append([llm_answer, correct_answer])

    IQ_Score = 0
    for item in IQ_results:
        if item[0] == item[1]:
            IQ_Score += 1
    print(round(IQ_Score / len(IQ_results), 3))

    EQ_Score = 0
    for item in EQ_results:
        if item[0] == item[1]:
            EQ_Score += 1
    print(round(EQ_Score / len(EQ_results), 3))

    total_score = 0
    for item in IQ_results + EQ_results:
        if item[0] == item[1]:
            total_score += 1
    print(round(total_score / (len(IQ_results) + len(EQ_results)), 3))