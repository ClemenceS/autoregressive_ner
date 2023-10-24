import argparse
from vllm import LLM, SamplingParams

argparse = argparse.ArgumentParser()
argparse.add_argument('--model', type=str, help='model name')

args = argparse.parse_args()


# def get_prompts_for_model(model_name, prompts):
#     if 'bloom' in model_name:
#         return prompts
#     from fastchat.model import get_conversation_template
#     prompts_for_model = []
#     for prompt in prompts:
#         conv = get_conversation_template(model_name)
#         conv.append_message(conv.roles[0], prompt)
#         conv.append_message(conv.roles[1], None)
#         prompts_for_model.append(conv.get_prompt())
#     return prompts_for_model

llm = LLM(args.model, tensor_parallel_size=2)
sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
)

s = """The task is to label all mentions of person names in a sentence. These are words that refer to the name of a real or fictional person. I can also put them in a specific format. Here are some examples of sentences I can handle:
Input: In 1995 , Barrymore starred in Boys on the Side opposite Whoopi Goldberg and Mary-Louise Parker , and had a cameo role in Joel Schumacher 's film Batman Forever , in which she portrayed a moll to Tommy Lee Jones ' character , Two-Face .
Output: In 1995 , @@Barrymore## starred in Boys on the Side opposite @@Whoopi Goldberg## and @@Mary-Louise Parker## , and had a cameo role in @@Joel Schumacher## 's film Batman Forever , in which she portrayed a moll to @@Tommy Lee Jones## ' character , @@Two-Face## .
Input: Ordoño III 's half-brother and successor , Sancho the Fat , had been deposed by his cousin Ordoño IV .
Output: @@Ordoño III## 's half-brother and successor , @@Sancho the Fat## , had been deposed by his cousin @@Ordoño IV## .
Input: In 1320 the Brandenburg Ascanian line came to an end , and from 1323 up until 1415 Brandenburg was under the control of the Wittelsbachs of Bavaria , followed by the Luxembourg dynasty .
Output: In 1320 the Brandenburg @@Ascanian## line came to an end , and from 1323 up until 1415 Brandenburg was under the control of the @@Wittelsbachs## of Bavaria , followed by the @@Luxembourg## dynasty .
Input: Charles was not as valued as his physically stronger , elder brother , Henry , Prince of Wales ; whom Charles personally adored and attempted to emulate .
Output: @@Charles## was not as valued as his physically stronger , elder brother , @@Henry## , Prince of Wales ; whom @@Charles## personally adored and attempted to emulate .
Input: Derleth wrote many and varied children 's works , including biographies meant to introduce younger readers to explorer Fr. Marquette , as well as Ralph Waldo Emerson and Henry David Thoreau .
Output: @@Derleth## wrote many and varied children 's works , including biographies meant to introduce younger readers to explorer @@Fr. Marquette## , as well as @@Ralph Waldo Emerson## and @@Henry David Thoreau## .
Imitate me. Identify all the mentions of person names in the following sentence, by putting "@@" in front and a "##" behind each of them.
Input: The new canal, commanded by Benjamin B. Odell Jr., replaced much of the original route , leaving many abandoned sections ( most notably between Syracuse and Rome ) .
Output: """

pre_outputs = llm.generate(s, sampling_params=sampling_params)
outputs = [o.outputs[0].text for o in pre_outputs]
print(outputs)
