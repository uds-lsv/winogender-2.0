from spacy.lang.en import English
import json
import csv

nlp = English()
tokenizer = nlp.tokenizer

def convert_to_jsonlines(data_file, out_file):
    with open(out_file, 'w', encoding='utf-8') as out_f:
        with open(data_file) as f:
            reader = csv.DictReader(f, delimiter='\t')

            for i,row in enumerate(reader):
                tokenized = tokenizer(row['sentence'])
                line = {'document_id': f'wb_{i}',
                        'cased_words': [t.text for t in tokenized],
                        'sent_id': [0] * len(tokenized)}
                line_str = json.dumps(line)
                out_f.write(line_str+'\n')

convert_to_jsonlines('double.tsv', 'double.jsonlines')
convert_to_jsonlines('single.tsv', 'single.jsonlines')
convert_to_jsonlines('double_old.tsv', 'double_old.jsonlines')
