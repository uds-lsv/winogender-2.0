import sys
import os
import json


def merge_subtokens(subtokens):
    words = []
    for subtoken in subtokens:
        if subtoken.startswith("##"):
            words[-1] = words[-1] + subtoken[2:]
        else:
            words.append(subtoken)
    return words


def convert_to_output_tsv(spanbert_out_file, tsv_out_file):
    with open(spanbert_out_file, "r") as f_spanbert_out, \
        open(tsv_out_file, "w") as f_tsv_out:
        f_tsv_out.write("occupation\tparticipant\tsentence\tpronoun_type\tpronoun\tanswer\tgeneration\n")
        for line in f_spanbert_out.read().split("\n"):
            if not line:
                continue
            line_dict = json.loads(line)
            sentence = line_dict["sentences"][-1]
            pronoun = line_dict["stimulus"]["pronoun"]
            entity_strings = []
            for cluster in line_dict["predicted_clusters"]:
                for entity in cluster:
                    start, end = entity
                    entity_strings.append(" ".join(merge_subtokens(sentence[start:end+1])))
                if pronoun not in entity_strings:
                    continue
                entity_strings.remove(pronoun)
                generation = entity_strings[0]
            f_tsv_out.write("\t".join(list(line_dict["stimulus"].values())) + "\t" + generation + "\n")


def main():
    for subdir in ["single", "double", "double_old"]:
        if not os.path.exists(subdir):
            os.mkdir(subdir)
    convert_to_output_tsv("spanbert/single_base_res.jsonlines", "single/spanbert_base.tsv")
    convert_to_output_tsv("spanbert/double_base_res.jsonlines", "double/spanbert_base.tsv")
    convert_to_output_tsv("spanbert/double_old_base_res.jsonlines", "double_old/spanbert_base.tsv")
    convert_to_output_tsv("spanbert/single_large_res.jsonlines", "single/spanbert_large.tsv")
    convert_to_output_tsv("spanbert/double_large_res.jsonlines", "double/spanbert_large.tsv")
    convert_to_output_tsv("spanbert/double_old_large_res.jsonlines", "double_old/spanbert_large.tsv")


if __name__ == "__main__":
    main()