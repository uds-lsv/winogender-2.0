import csv
from fastcoref import LingMessCoref

model = LingMessCoref(device='cuda:0')

def evaluate_lingmess(data_file, out_file):
    print(data_file, out_file)
    with open(out_file, 'w', encoding='utf-8') as out_f:
        with open(data_file, 'r', encoding='utf-8') as f:
            data_reader = csv.DictReader(f, delimiter='\t')
            out_f.write('\t'.join(data_reader.fieldnames + ['generation']) + '\n')

            for row in data_reader:
                preds = model.predict(texts=[row['sentence']])
                clusters = preds[0].get_clusters()
                generation = ''
                for entity_strings in clusters:
                    if not row['pronoun'] in entity_strings:
                        # we don't care about demonstration-it coreferences, for example
                        continue
                    entity_strings.remove(row['pronoun'])
                    generation = entity_strings[0]
                out_f.write('\t'.join(list(row.values()) + [generation]) + '\n')

evaluate_lingmess('double.tsv', 'double/lingmess.tsv')
evaluate_lingmess('single.tsv', 'single/lingmess.tsv')
evaluate_lingmess('double_old.tsv', 'double_old/lingmess.tsv')
