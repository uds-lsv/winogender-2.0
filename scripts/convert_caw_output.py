import json
import csv

def convert_to_output_tsv(data_file, caw_coref_out_file, tsv_out_file):
    print(data_file, caw_coref_out_file, tsv_out_file)
    with open(tsv_out_file, 'w', encoding='utf-8') as tsv_out_f:
        with open(data_file, 'r', encoding='utf-8') as f:
            with open(caw_coref_out_file, 'r', encoding='utf-8') as caw_coref_out_f:
                data_reader = csv.DictReader(f, delimiter='\t')
                tsv_out_f.write('\t'.join(data_reader.fieldnames + ['generation']) + '\n')

                for i, (row, line) in enumerate(zip(data_reader, caw_coref_out_f)):
                    line_dict = json.loads(line)
                    assert f'wb_{i}' == line_dict['document_id']
                    generation = ''
                    for cluster in line_dict['span_clusters']:
                        entity_strings = []
                        for entity in cluster:
                            start, end = entity[0], entity[1]
                            # a hack, but we can get away with it
                            entity_strings.append(' '.join(line_dict['cased_words'][start:end]))
                        if not row['pronoun'] in entity_strings:
                            # we don't care about demonstration-it coreferences, for example
                            continue
                        entity_strings.remove(row['pronoun'])
                        generation = entity_strings[0]
                    tsv_out_f.write('\t'.join(list(row.values()) + [generation]) + '\n')

convert_to_output_tsv('double.tsv', 'double_output.jsonlines', 'double/caw_coref.tsv')
convert_to_output_tsv('single.tsv', 'single_output.jsonlines', 'single/caw_coref.tsv')
convert_to_output_tsv('double_old.tsv', 'double_old_output.jsonlines', 'double_old/caw_coref.tsv')
