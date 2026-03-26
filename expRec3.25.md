#  数据映射
- delta方法在capability tree的投影
#  delta origin
export DEEPSEEK_API_KEY="sk-ab412f420cd540888da4732a35600c4a"
python src/4.2/delta_origin.py   --data_path data/banking77/train.jsonl   --output_path data/banking77/train_delta_origin_scored.jsonl --max_samples 500 --concurrancy 32