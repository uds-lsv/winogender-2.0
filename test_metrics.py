from scripts.compute_performance import Scorer
import pandas as pd


def test_verbalized_matches_answer_not_distractor():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician", "technician", "technician"],
            "participant": ["customer", "customer", "customer", "customer"],
            "pronoun": ["he", "she", "they", "xe"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["customer", "customer", "customer", "customer"],
            "generation": ["customer", "customer", "customer", "customer"]
        }
    )

    scorer = Scorer(df)

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    assert tp == 4
    assert tn == 4
    assert fp == 0
    assert fn == 0
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    assert f1 == 1.0
    assert recall == 1.0
    assert precision == 1.0
    assert accuracy == 1.0


def test_verbalized_matches_distractor_not_answer():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician", "technician", "technician"],
            "participant": ["customer", "customer", "customer", "customer"],
            "pronoun": ["he", "she", "they", "xe"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["customer", "customer", "customer", "customer"],
            "generation": ["technician", "technician", "technician", "technician"]
        }
    )

    scorer = Scorer(df)

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    assert tp == 0
    assert tn == 0
    assert fp == 4
    assert fn == 4
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    assert f1 == 0.0
    assert recall == 0.0
    assert precision == 0.0
    assert accuracy == 0.0
    

def test_verbalized_matches_neither_distractor_not_answer():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician", "technician", "technician"],
            "participant": ["customer", "customer", "customer", "customer"],
            "pronoun": ["he", "she", "they", "xe"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["customer", "customer", "customer", "customer"],
            "generation": ["", "", "", ""]
        }
    )

    scorer = Scorer(df)

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    assert tp == 0
    assert tn == 4
    assert fp == 0
    assert fn == 4
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    assert f1 == 0.0
    assert recall == 0.0
    assert precision == 1.0
    assert accuracy == 0.0
    

def test_2_abstain_2_correct():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician", "technician", "technician"],
            "participant": ["customer", "customer", "customer", "customer"],
            "pronoun": ["he", "she", "they", "xe"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["customer", "customer", "customer", "customer"],
            "generation": ["customer", "", "customer", ""]
        }
    )

    scorer = Scorer(df)

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    print(tp, tn, fp, fn)
    assert tp == 2
    assert tn == 4
    assert fp == 0
    assert fn == 2
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    print(f1, recall, precision, accuracy)
    assert f1 == 2/3
    assert recall == 0.5
    assert precision == 1.0
    assert accuracy == 0.5


def test_2_abstain_2_incorrect():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician", "technician", "technician"],
            "participant": ["customer", "customer", "customer", "customer"],
            "pronoun": ["he", "she", "they", "xe"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["customer", "customer", "customer", "customer"],
            "generation": ["technician", "", "technician", ""]
        }
    )

    scorer = Scorer(df)

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    print(tp, tn, fp, fn)
    assert tp == 0
    assert tn == 2
    assert fp == 2
    assert fn == 4
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    print(f1, recall, precision, accuracy)
    assert f1 == 0.0
    assert recall == 0.0
    assert precision == 0.0
    assert accuracy == 0.0


def test_paired_4_correct():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician", "technician", "technician"],
            "participant": ["customer", "customer", "customer", "customer"],
            "pronoun": ["he", "she", "they", "xe"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["customer", "customer", "customer", "customer"],
            "generation": ["customer", "customer", "customer", "customer"]
        }
    )

    scorer = Scorer(df)
    scorer.pairing = "template"

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    print(tp, tn, fp, fn)
    assert tp == 4
    assert tn == 4
    assert fp == 0
    assert fn == 0
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    print(f1, recall, precision, accuracy)
    assert f1 == 1.0
    assert recall == 1.0
    assert precision == 1.0
    assert accuracy == 1.0


def test_paired_1_correct_3_incorrect():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician", "technician", "technician"],
            "participant": ["customer", "customer", "customer", "customer"],
            "pronoun": ["he", "she", "they", "xe"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["customer", "customer", "customer", "customer"],
            "generation": ["technician", "customer", "customer", "customer"]
        }
    )

    scorer = Scorer(df)
    scorer.pairing = "template"

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    print(tp, tn, fp, fn)
    assert tp == 0
    assert tn == 0
    assert fp == 4
    assert fn == 4
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    print(f1, recall, precision, accuracy)
    assert f1 == 0.0
    assert recall == 0.0
    assert precision == 0.0
    assert accuracy == 0.0


def test_coref_dir_2_correct():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician"],
            "participant": ["customer", "customer"],
            "pronoun": ["he", "he"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["technician", "customer"],
            "generation": ["technician", "customer"]
        }
    )

    scorer = Scorer(df)
    scorer.pairing = "coref_dir"

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    print(tp, tn, fp, fn)
    assert tp == 2
    assert tn == 2
    assert fp == 0
    assert fn == 0
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    print(f1, recall, precision, accuracy)
    assert f1 == 1.0
    assert recall == 1.0
    assert precision == 1.0
    assert accuracy == 1.0


def test_coref_dir_1_correct_1_incorrect():
    df = pd.DataFrame(
        {
            "occupation": ["technician", "technician"],
            "participant": ["customer", "customer"],
            "pronoun": ["he", "he"],
            "pronoun_type": ["$NOM_PRONOUN", "$NOM_PRONOUN"],
            "answer": ["technician", "customer"],
            "generation": ["technician", "technician"]
        }
    )

    scorer = Scorer(df)
    scorer.pairing = "coref_dir"

    tp, tn, fp, fn = scorer.get_tp(), scorer.get_tn(), scorer.get_fp(), scorer.get_fn()
    print(tp, tn, fp, fn)
    assert tp == 0
    assert tn == 0
    assert fp == 2
    assert fn == 2
    assert tp + fp + fn + tn == 2*len(df)
    
    f1, recall, precision, accuracy = scorer.get_f1(), scorer.get_recall(), scorer.get_precision(), scorer.get_accuracy()
    print(f1, recall, precision, accuracy)
    assert f1 == 0.0
    assert recall == 0.0
    assert precision == 0.0
    assert accuracy == 0.0
