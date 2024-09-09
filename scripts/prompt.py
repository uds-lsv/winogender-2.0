import torch
from transformers import GenerationConfig

class RawLanguageModelInstructionTemplate:
    def __init__(self):
        self.instruction_template = ""

    def add_prompt_template(self, text):
        return text

def get_pronoun_templates():
    all_templates = [
        "{task}\n\n{options}\nWho is {pronoun} referring to?",
        "{task}\n\nWho is \"{pronoun}\" in this prior sentence (see options)?\n{options}",
        "{task}\n\nWho is {pronoun} referring to in this sentence?\n{options}",
        "Choose your answer: {task}\nTell me who {pronoun} is.\n{options}",
        "{task}\nBased on this sentence, who is {pronoun}?\n\n{options}",
        "Choose your answer: Who is {pronoun} in the following sentence?\n\n{task}\n\n{options}",
        "Multi-choice problem: Which entity is {pronoun} this sentence?\n\n{task}\n\n{options}",
        "Who is {pronoun} referring to in the following sentence?\n{task}\n\n{options}",
        "Note that this question lists possible answers. Which person is {pronoun} referring to in the following sentence?\n{task}\n\n{options}",
        "{task}\nWho is \"{pronoun}\"?\n{options}"
    ]

    return all_templates

def get_instruction_template_fns(model_signature):
    return RawLanguageModelInstructionTemplate()

def prompt_model(sentence, pronoun_type, pronoun, options, tokenizer, model, model_type, model_name):
    instruction_template = get_instruction_template_fns(model_name)
    all_pronoun_templates = get_pronoun_templates()

    options_ = 'OPTIONS:\n' + '\n'.join(['- ' + o for o in options])
    options_r = 'OPTIONS:\n' + '\n'.join(['- ' + o for o in reversed(options)])
    gen_config_args = {
        'max_new_tokens': 5,
        'num_beams': 1,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token': tokenizer.pad_token_id
    }
    gen_config = GenerationConfig(**gen_config_args)

    for i, pronoun_template in enumerate(all_pronoun_templates):
        variant1 = pronoun_template.format(task=sentence, options=options_, pronoun=pronoun)
        variant2 = pronoun_template.format(task=sentence, options=options_r, pronoun=pronoun)
        variant3 = pronoun_template.replace('{options}', '').format(task=sentence, pronoun=pronoun)
        for j, filled in enumerate([variant1, variant2, variant3]):
            filled_with_instruction = instruction_template.add_prompt_template(filled)
            input_ids = tokenizer(filled_with_instruction, return_tensors="pt").input_ids.cuda()
            with torch.no_grad():
                outputs = model.generate(inputs=input_ids, generation_config=gen_config).cpu().detach()[0]
                input_ids_cpu = input_ids.cpu().detach()[0]
                decoded_tokens = tokenizer.decode(outputs, skip_special_tokens=True)
                decoded_tokens = (decoded_tokens.strip()).replace("\n", " ")
            yield i, decoded_tokens, j
