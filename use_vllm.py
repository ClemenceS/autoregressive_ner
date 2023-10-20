import argparse
from vllm import LLM, SamplingParams

argparse = argparse.ArgumentParser()
argparse.add_argument('--model', type=str, help='model name')

args = argparse.parse_args()

llm = LLM(args.model, tensor_parallel_size=2)
sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
)
texts = [
    "The answer to life, the universe and everything is",
]
pre_outputs = llm.generate(texts, sampling_params)
outputs = [o.outputs[0].text for o in pre_outputs]
print(outputs)
