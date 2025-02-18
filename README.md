# note-r1

存放 deepseek-r1 等一些强化学习脚本，暂且用于存放蒸馏比赛脚本

## result

> Setting: 
> - max_length: 4096
> - do_sample: False
> - top_k: 50
> - top_p: 1.0
> - temperature: 1.0
> - max_new_tokens: 4096

| Model                         | IQ       | EQ       | Score            |
|-------------------------------|----------|----------|------------------|
| Qwen2.5-7B-Instruct           | 0.50     | 0.6125   | 0.575            |
| Qwen2.5-14B-Instruct          | 0.525    | 0.75     | 0.6750           |
| Qwen2.5-32B-Instruct          | 0.650    | 0.7875   | 0.7417           |
| DeepSeek-R1-Distill-Qwen-7B   | 0.675    | 0.30     | 0.425            |
| DeepSeek-R1-Distill-Qwen-14B  | 0.625    | 0.725    | 0.6917           |