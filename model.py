import json
from tqdm import tqdm
from openai import OpenAI
from openai import APITimeoutError
import os
import time


class APIModel:
    def __init__(self, model, api_key, base_url):
        self.__api_key = api_key
        self.__base_url = base_url
        self.model = model
        self.client = OpenAI(api_key=self.__api_key, base_url=self.__base_url)

    def generate_prompt(self, data):
        question = data['question']
        options = [f"({chr(65+i)}): {option}" for i, option in enumerate(data['choices'])]
        options_str = '\n'.join(options)
        prompt = f"问题: {question}\n{options_str}\n"
        return prompt

    def get_compention(self, prompt, system=None):
        if system is None:
            system = 'You are a helpful assistant.'
        
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {'role': 'system', 'content': system},
                        {'role': 'user', 'content': prompt}
                    ],
                    stream=False,
                    temperature=0.6,
                )
                reasoning_content = response.choices[0].message.reasoning_content
                content = response.choices[0].message.content
                return f"<think>\n{reasoning_content}</think>\n{content}"
            
            except APITimeoutError:
                retry_count += 1
                if retry_count >= max_retries:
                    raise Exception(f"API请求超时，已达最大重试次数（{max_retries}次）")
                print(f"请求超时，正在重试 ({retry_count}/{max_retries})...")
                time.sleep(1)  # 简单退避策略
                
            except Exception as e:
                # 其他异常直接抛出
                raise e
        
        raise Exception("未知错误，重试逻辑异常")  # 理论上不会执行到此处
    
if __name__ == "__main__":
    # 初始化 APIModel 实例
    model = APIModel(
        model="Pro/deepseek-ai/DeepSeek-R1", 
        api_key='xxx',
        base_url='https://api.siliconflow.cn/v1/'
    )
    system_prompt = "你是一个高智商和高情商的专家，你被要求回答一个选择题，并选出一个正确的选项，解释原因，最终输出格式为：`答案是(选项)`。"
    # 单次交互示例
    with open('./dataset/EQ-IQ.jsonl', 'r', encoding='utf-8') as f:
        data = f.readlines()
    for i in tqdm(range(0, len(data))):
        item = json.loads(data[i])
        question_prompt = model.generate_prompt(item)
        response = model.get_compention(question_prompt, system_prompt)
        with open('distill-EQ-IQ.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps({
                'system': system_prompt,
                'input': "",
                'instruction': question_prompt,
                'output': response,
                'correct': item['answer']
            }, ensure_ascii=False) + '\n')
        time.sleep(10)
    