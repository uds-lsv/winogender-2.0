import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration
import numpy as np
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict
from pathlib import Path
from pronouns import mapping
from prompt import prompt_model
import csv
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(model_name, model_type):
    return T5ForConditionalGeneration.from_pretrained(model_name,
            torch_dtype=torch.float16, device_map='auto')

def get_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name)

# ordered by number of parameters
models = [
    ('google/flan-t5-small', 'enc-dec'), # 77M
    ('google/flan-t5-base', 'enc-dec'), # 248M
    ('google/flan-t5-large', 'enc-dec'), # 783M
    ('google/flan-t5-xl', 'enc-dec'), # 2.85B
    ('google/flan-t5-xxl', 'enc-dec'), # 11.3B
]

def construct_model_file_map(input_files):
    model_file_map = defaultdict(list)
    for data_file in input_files:
        # make directory for results
        stem = Path(data_file).stem
        folder = Path(stem)
        folder.mkdir(exist_ok=True)
        for MODEL, model_type in models:
            prompt_out_file = Path(folder / f"prompt_{MODEL.replace('/', '_')}.tsv")
            if prompt_out_file.exists():
                continue
            else:
                model_file_map[MODEL].append((model_type, data_file, prompt_out_file))
    return model_file_map

def main():
    assert len(sys.argv) >= 2

    model_file_map = construct_model_file_map(sys.argv[1:])

    for MODEL in model_file_map:
        print(f'loading {MODEL}')
        model_type = model_file_map[MODEL][0][0]
        model = get_model(MODEL, model_type)
        tokenizer = get_tokenizer(MODEL)
        model.eval() # disable dropout
        for model_type, data_file, out_file in model_file_map[MODEL]:
            print(data_file, out_file)
            prompt_header = [
                'sentence',
                'generation', # new
                'pronoun_type',
                'occupation',
                'participant',
                'pronoun',
                'answer',
                'prompt', # new
                'options', # new
            ]
            with open(out_file, 'w', encoding='utf-8') as prompt_out_f:
                with open(data_file) as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    prompt_out_f.write('\t'.join(prompt_header) + '\n')

                    for row in reader:
                        options = [row['occupation'], row['participant']]
                        for prompt, generation, opts_config in prompt_model(row['sentence'], row['pronoun_type'], row['pronoun'], options, tokenizer, model, model_type, MODEL):
                            data = [
                                row['sentence'],
                                generation,
                                row['pronoun_type'],
                                row['occupation'],
                                row['participant'],
                                row['pronoun'],
                                row['answer'],
                                str(prompt),
                                str(opts_config)
                            ]
                            prompt_out_f.write('\t'.join(data) + '\n')

if __name__ == '__main__':
    main()
