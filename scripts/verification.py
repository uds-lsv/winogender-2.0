from pronouns import mapping
import csv
import sys
import itertools

pronouns = ['he', 'she', 'they', 'xe',
            'him', 'her', 'them', 'xem',
            'his', 'her', 'their', 'xyr',
            'hers', 'theirs', 'xyrs',
            'himself', 'herself', 'themself',
            'theirself', 'themselves', 'theirselves',
            'xemself', 'xyrself']

# adapted from https://docs.python.org/3/library/itertools.html
def batched(iterable, n):
    # batched('ABCDEFGH', 2) --> AB CD EF GH
    # batched('ABCDEFG', 2) --> AB CD EF AssertionError!
    if n < 1:
        raise ValueError('n must be at least one')
    while batch := tuple(itertools.islice(iter(iterable), n)):
        assert len(batch) == n
        yield batch

def replace_pronouns(template, mapping):
    for case in mapping:
        if case in template:
            for pronoun in mapping[case]:
                yield (pronoun, template.replace(case, pronoun))

def replace_entity_placeholders(row, which='double'):
    if which == 'double':
        key = 'template'
    else:
        key = 'single_version'
    return row[key].strip().replace(
        '$OCCUPATION', row['occupation(0)']).replace(
        '$PARTICIPANT', row['other-participant(1)'])

def test_case_agnostic_assertions(row1, row2, pronoun_type, occupation, participant):
    try:
        assert row1['pronoun_type'] == row2['pronoun_type'] == pronoun_type
    except AssertionError:
        raise AssertionError(f'mismatch in pronoun type ({pronoun_type}) with {occupation} and {participant}') from e

    # check that the templates exist
    template1, template2 = replace_entity_placeholders(row1), replace_entity_placeholders(row2)
    single_template1, single_template2 = replace_entity_placeholders(row1, 'single'), replace_entity_placeholders(row2, 'single')
    try:
        assert template1
        assert template2
        assert single_template1
        assert single_template2
    except AssertionError as e:
        raise AssertionError(f'missing templates for {pronoun_type}') from e

    # check that the right pronouns are in the template
    for template in [template1, template2, single_template1, single_template2]:
        try:
            assert pronoun_type in template
        except AssertionError as e:
            raise AssertionError(f'missing {pronoun_type} in template {template}') from e

    # check that there are no other pronouns in the template
    for template in [template1, template2, single_template1, single_template2]:
        try:
            for tok in template.lower().split():
                assert tok not in pronouns
        except AssertionError as e:
            raise AssertionError(f'extra pronoun in template: {template.lower()}') from e

    # check that the prefixes up to the pronoun match exactly
    try:
        prefix1, prefix2 = template1.split(pronoun_type)[0], template2.split(pronoun_type)[0]
        assert prefix1 == prefix2
    except AssertionError as e:
        raise AssertionError(f'prefix mismatch: {prefix1} != {prefix2}') from e
        
    # check that "the $OCCUPATION" and "the $PARTICIPANT" are present in both templates
    try:
        assert f'the {occupation}' in prefix1.lower() and f'the {participant}' in prefix1.lower()
    except AssertionError as e:
        raise AssertionError(f'missing the: {prefix1.lower()}') from e

    # check that "the $OCCUPATION" and "the $PARTICIPANT" are present in the single-entity templates
    assert (row1['answer'] == '0' and row2['answer'] == '1') or (row1['answer'] == '1' and row2['answer'] == '0')
    if row1['answer'] == '0':
        try:
            assert f'the {occupation}' in single_template1.lower()
        except AssertionError as e:
            raise AssertionError(f'missing the: {single_template1.lower()}') from e
    elif row1['answer'] == '1':
        try:
            assert f'the {participant}' in single_template1.lower()
        except AssertionError as e:
            raise AssertionError(f'missing the: {single_template1.lower()}') from e
    if row2['answer'] == '0':
        try:
            assert f'the {occupation}' in single_template2.lower()
        except AssertionError as e:
            raise AssertionError(f'missing the: {single_template2.lower()}') from e
    elif row2['answer'] == '1':
        try:
            assert f'the {participant}' in single_template2.lower()
        except AssertionError as e:
            raise AssertionError(f'missing the: {single_template2.lower()}') from e

def test_batch(batch):
    occupations = [row['occupation(0)'] for row in batch]
    participants = [row['other-participant(1)'] for row in batch]
    try:
        assert len(set(occupations)) == 1
    except AssertionError as e:
        raise AssertionError(f'inconsistent occupations: {occupations}') from e
    try:
        assert len(set(participants)) == 1
    except AssertionError as e:
        raise AssertionError(f'inconsistent participants: {participants}') from e
    
    test_case_agnostic_assertions(batch[0], batch[1], '$NOM_PRONOUN', occupations[0], participants[0])
    test_case_agnostic_assertions(batch[2], batch[3], '$ACC_PRONOUN', occupations[0], participants[0])
    test_case_agnostic_assertions(batch[4], batch[5], '$POSS_PRONOUN', occupations[0], participants[0])

def main():
    assert len(sys.argv) == 2
    with open(sys.argv[1]) as f, \
            open('automatically_validated.tsv', 'w', encoding='utf-8') as f_double_human, \
            open('automatically_validated_singles.tsv', 'w', encoding='utf-8') as f_single_human, \
            open('double.tsv', 'w', encoding='utf-8') as f_double, \
            open('single.tsv', 'w', encoding='utf-8') as f_single:
        header = ['occupation', 'participant', 'sentence', 'pronoun_type', 'pronoun', 'answer']
        f_double.write('\t'.join(header) + '\n')
        f_single.write('\t'.join(header) + '\n')

        reader = csv.DictReader(f, delimiter='\t')
        for batch in batched(reader, 6):
            # check that the occupations and the participants match for the entire batch
            # (otherwise something is very wrong with your data)
            try:
                test_batch(batch)
            except AssertionError as e:
                raise AssertionError from e
            else:
                for row in batch:
                    label = row['occupation(0)']
                    other = row['other-participant(1)']
                    if row['answer'] == '1':
                        label = row['other-participant(1)']
                        other = row['occupation(0)']
                    template = row['template'].strip()
                    entity_replaced = replace_entity_placeholders(row, 'double')
                    pronoun_type = row['pronoun_type']
                    assert pronoun_type in mapping

                    # double-entity templates
                    # human validation file
                    for pronoun, sentence in replace_pronouns(entity_replaced, mapping):
                        f_double_human.write(f'{template}\t{sentence}\tDoes the pronoun {pronoun} refer to the {label}?\n')
                        f_double_human.write(f'{template}\t{sentence}\tDoes the pronoun {pronoun} refer to the {other}?\n')
                        # final data file
                        data = [row['occupation(0)'], row['other-participant(1)'], sentence, pronoun_type, pronoun, label]
                        f_double.write('\t'.join(data) + '\n')

                    # single-entity templates
                    # human validation files
                    template = row['single_version'].strip()
                    entity_replaced = replace_entity_placeholders(row, 'single')
                    for pronoun, sentence in replace_pronouns(entity_replaced, mapping):
                        f_single_human.write(f'{template}\t{sentence}\tDoes the pronoun {pronoun} refer to the {label}?\n')
                       # final data file
                        data = [row['occupation(0)'], row['other-participant(1)'], sentence, pronoun_type, pronoun, label]
                        f_single.write('\t'.join(data) + '\n')

if __name__ == '__main__':
    main()
