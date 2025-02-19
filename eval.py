from evalscope.run import run_task
from evalscope.config import TaskConfig

"""
siliconflow: https://api.siliconflow.cn/v1/chat/completions
dashscope: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
modelscope: https://api-inference.modelscope.cn/v1/chat/completions
xunfei: https://maas-api.cn-huabei-1.xf-yun.com/v1/chat/completions
"""

task_cfg = TaskConfig(
    model='Qwen/Qwen2.5-72B-Instruct',
    api_url='https://api.siliconflow.cn/v1/chat/completions',
    api_key='xxxx',
    eval_type='service',
    datasets=['iquiz'],
    generation_config={
        'max_tokens': 4096,
        'max_new_tokens': 4096,
    }
)

run_task(task_cfg=task_cfg)