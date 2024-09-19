# Revisiting English Winogender Schemas for Consistency, Coverage, and Grammatical Case

This repository contains code and data for our paper, which presents WinoPron, an expanded dataset that also fixes issues with Winogender Schemas (Rudinger et al., 2018).

## Data

We provide our newly constructed templates in a zipped data file. Please unzip it with the password `vogelbeobachtung131719`, using a command like `unzip -P PASSWORD FILE.zip`.
It contains two files:
* `new_templates.tsv`: WinoPron templates, where each row contains a sentence for a given occupation, participant and pronoun type, as well as
  a single-entity sentence that contains a version of the sentence with similar content but only one of the entities (either the occupation or participant)
  such that the coreference is maintained. The answer column is either 0 for a coreference with the occupation or 1 for coreference with the participant.
* `old_templates.tsv`: Winogender schemas (Rudinger et al., 2018) templates for comparison, where each row contains a sentence for a given occupation, participant and pronoun type;
  the answer column is either 0 for a coreference with the occupation or 1 for coreference with the participant.

After unzipping, run:

`python3 scripts/verification.py new_templates.tsv` to verify the templates with some manual checks, which creates 4 files:
`automatically_validated_singles.tsv` and `automatically_validated.tsv` for human validation with Google sheets; and
`double.tsv` and `single.tsv` with single- and double-entity sentences for evaluation.

Run `python3 scripts/prep_old_templates.py old_templates.tsv` to instantiate and convert the Winogender schemas templates into 
`double_old.tsv`, which is in the same format as our WinoPron templates.

## Running FLAN-T5

`python3 scripts/score_models.py double.tsv single.tsv double_old.tsv` (uses `scripts/prompt.py` behind the scenes)
Evaluates FLAN-T5 models of five sizes on the data files with 10 x 3 prompts and stores results in subfolders named after the data files.

## Running SpanBERT

`python3 scripts/spanbert_inputs.py path/to/stimuli path/to/vocab/file`
Converts stimuli into jsonlines format for SpanBERT and creates two files: `single.jsonlines` and `double.jsonlines`

## Running CAW-coref

`python3 scripts/convert_for_caw.py`
Converts `double.tsv`, `single.tsv` and `double_old.tsv` to jsonlines format for the CAW-coref system.

Follow the setup instructions in the [CAW-coref repository](https://github.com/KarelDO/wl-coref) and produce output files named `single_output.jsonlines`, `double_output.jsonlines`, and `double_old_output.jsonlines`.

`python3 scripts/convert_caw_output.py`
Converts `single_output.jsonlines`, `double_output.jsonlines`, and `double_old_output.jsonlines` into TSV output files in subfolders named after the data files.

## Running LingMess

`python3 scripts/evaluate_lingmess.py`
Evaluates LingMess on the data files and stores TSV results in subfolders named after the data files.

## Compute performance

`python3 scripts/compute_performance.py`
Computes accuracy for FLAN-T5, CAW-coref and LingMess on WinoPron (double-entity and single-entity versions), and Winogender schemas (Rudinger et al., 2018).
