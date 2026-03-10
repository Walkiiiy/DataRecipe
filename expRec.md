- 数据集tag
```
OPENAI_API_KEY='sk-ab412f420cd540888da4732a35600c4a' OPENAI_BASE_URL='https://api.deepseek.com/v1' python src/4.1/stage1_atomic_profile.py   --model deepseek-chat   --max-samples 1000   --batch-size 8   --concurrency 10   --max-tokens 700   --output data/alpaca_with_tags.jsonl
```
- 