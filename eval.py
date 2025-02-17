from evalscope.run import run_task
from evalscope.config import TaskConfig

task_cfg = TaskConfig(
    model='deepseek-r1',
    api_url='https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    api_key='sk-26dbb562231740ef92f004ed10a41ab5',
    eval_type='service',
    datasets=['iquiz'],
    generation_config={
        'max_tokens': 8192,
    }
)

run_task(task_cfg=task_cfg)