from verification import batched, replace_pronouns
from pronouns import mapping
import csv
import sys
import itertools

def replace_entity_placeholders(row):
    key = 'sentence'
    return row[key].strip().replace(
        '$OCCUPATION', row['occupation(0)']).replace(
        '$PARTICIPANT', row['other-participant(1)'])

def main():
    assert len(sys.argv) == 2
    with open(sys.argv[1]) as f, \
            open('double_old.tsv', 'w', encoding='utf-8') as f_double:
        header = ['occupation', 'participant', 'sentence', 'pronoun_type', 'pronoun', 'answer']
        f_double.write('\t'.join(header) + '\n')

        reader = csv.DictReader(f, delimiter='\t')
        for batch in batched(reader, 2):
            for row in batch:
                label = row['occupation(0)']
                other = row['other-participant(1)']
                if row['answer'] == '1':
                    label = row['other-participant(1)']
                    other = row['occupation(0)']
                template = row['sentence'].strip()
                entity_replaced = replace_entity_placeholders(row)
                pronoun_type = list(filter(lambda x: len(x) > 0,
                    list(map(lambda x : x if x in template else '', mapping.keys()))))[0]
                assert pronoun_type in mapping

                # double-entity templates
                for pronoun, sentence in replace_pronouns(entity_replaced, mapping):
                    # final data file
                    data = [row['occupation(0)'], row['other-participant(1)'],
                            sentence.replace('they was', 'they were'),
                            pronoun_type, pronoun, label]
                    f_double.write('\t'.join(data) + '\n')

if __name__ == '__main__':
    main()
