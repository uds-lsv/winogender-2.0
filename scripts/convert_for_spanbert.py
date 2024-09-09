import sys
import os
import csv
import json

from transformers import BertTokenizer


def create_subtoken_map(subtokens):
    word_indices = []
    current_index = -1

    for subword in subtokens:
        if subword == "[CLS]":
            word_indices.append(0)
        elif subword == "[SEP]":
            word_indices.append(current_index)
        elif subword.startswith("##"):
            word_indices.append(current_index)
        else:
            current_index += 1
            word_indices.append(current_index)

    return word_indices


def get_word_spans(subtokens):
    word_spans = {}
    curr_word = subtokens[0]
    curr_subtoken_index = 0
    subtoken_indices = [0]

    for subtoken in subtokens[1:]:
        if subtoken.startswith("##"):
            curr_word += subtoken[2:]
            subtoken_indices.append(curr_subtoken_index)
        else:
            word_spans[curr_word] = subtoken_indices if len(subtoken_indices) > 1 \
                else [curr_subtoken_index, curr_subtoken_index]
            curr_word = subtoken
            subtoken_indices = [curr_subtoken_index]
        curr_subtoken_index += 1

    return word_spans


def create_jsonlines_dicts(stimuli, tokenizer):

    for stimulus in stimuli:
        tokenized = tokenizer(stimulus["sentence"])
        subtokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
        # pronoun_cluster = get_word_spans(subtokens)[stimulus["pronoun"]]
        speakers = ["[SPL]"] + ["-" for _ in range(len(subtokens)-2)] + ["[SPL]"]
        sentence_map = [0 for _ in range(len(subtokens))]
        subtoken_map = create_subtoken_map(subtokens)
        assert len(speakers) == len(subtokens)
        assert len(subtoken_map) == len(subtokens)
        jsonlines_dict = {}
        # jsonlines_dict["clusters"] = [[pronoun_cluster]]
        jsonlines_dict["clusters"] = []
        jsonlines_dict["doc_key"] = "dw"
        jsonlines_dict["sentences"] = [subtokens]
        jsonlines_dict["speakers"] = [speakers]
        jsonlines_dict["subtoken_map"] = subtoken_map
        jsonlines_dict["sentence_map"] = sentence_map
        jsonlines_dict["stimulus"] = stimulus

        yield jsonlines_dict

def convert_to_jsonlines(data_file, out_file):
    with open(data_file, "r") as f_data, open(out_file, "w") as f_out:
        for jsonlines_dict in create_jsonlines_dicts(csv.DictReader(f_data, delimiter="\t"), tokenizer):
            json.dump(jsonlines_dict, f_out)
            f_out.write("\n")


if __name__ == "__main__":
    # input_dir = sys.argv[1] # The path to the single/double antecedent stimuli
    bert_vocab_file_path = sys.argv[1] # The path to the SpanBERT vocab file (in the SpanBERT repo)
    assert bert_vocab_file_path.endswith(".txt")
    tokenizer = BertTokenizer(bert_vocab_file_path, do_lower_case=False)

    if not os.path.exists("spanbert"):
        os.mkdir("spanbert")
    convert_to_jsonlines("single.tsv", "spanbert/single.jsonlines")
    convert_to_jsonlines("double.tsv", "spanbert/double.jsonlines")
    convert_to_jsonlines("double_old.tsv", "spanbert/double_old.jsonlines")
