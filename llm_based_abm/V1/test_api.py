"""运行 python test_api.py；测试 Pro 时在末尾加 doubao-seed-2-1-pro-260915。"""
import sys
from openai import OpenAI

model = sys.argv[1] if len(sys.argv) > 1 else 'ep-20261010131845-ps8n8'

with open('../key.txt', encoding='utf-8') as f:
    api_key = f.read().strip().strip('\"\'')

client = OpenAI(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
    timeout=30,
    max_retries=0,
)

# 一次短请求，不搜索、不携带历史，并关闭深度思考。
response = client.responses.create(
    model=model,
    input='只回复OK',
    max_output_tokens=16,
    extra_body={'thinking': {'type': 'disabled'}},
)

print('模型：', model)
print('回复：', response.output_text)
print('状态：', response.status)
print('输入 token：', response.usage.input_tokens)
print('输出 token：', response.usage.output_tokens)
print('合计 token：', response.usage.total_tokens)
