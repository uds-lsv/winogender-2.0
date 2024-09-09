import pandas as pd
import os
import sys

class Scorer(object):

    def __init__(self, df):
        self._df = self._preprocess_df(df)
        self._filtering = None
        self._pairing = None

    def _preprocess_df(self, df):
        def _extract_answer(row):
            generation, occupation, participant = row.generation, row.occupation, row.participant
            words = set([w.lower().replace('"', '').replace('.', '') for w in str(generation).split()])
            extracted_prons = list(words.intersection([occupation, participant]))
            if len(extracted_prons) == 1:
                return extracted_prons[0]
            return None
        df['verbalized_token'] = df.apply(_extract_answer, axis=1)
        df['distractor'] = df.apply(lambda row: row.participant if row.answer == row.occupation else row.occupation, axis=1)
        df["correct"] = df['correct'] = df['verbalized_token'] == df['answer']
        # add template information
        df_template = df.groupby(['occupation', 'participant', 'pronoun_type', 'answer'])["correct"].sum().reset_index().drop("correct", axis=1)
        df_template["template"] = list(range(len(df_template)))        
        df = pd.merge(df, df_template, on=['occupation', 'participant', 'pronoun_type', 'answer'])
        # add coref direction information
        df_coref_dir = df.groupby(['occupation', 'participant', 'pronoun', 'pronoun_type'])["correct"].sum().reset_index().drop("correct", axis=1)
        df_coref_dir["coref_dir"] = list(range(len(df_coref_dir)))
        df = pd.merge(df, df_coref_dir, on=['occupation', 'participant', 'pronoun', 'pronoun_type'])
        return df
    
    def _filter_df(self, df):
        if self.filtering == None:
            return df
        if self.filtering == 'nom':
            filtered_df = df[df['pronoun_type'] == '$NOM_PRONOUN']
        elif self.filtering == 'acc':
            filtered_df = df[df['pronoun_type'] == '$ACC_PRONOUN']
        elif self.filtering == 'poss':
            filtered_df = df[df['pronoun_type'] == '$POSS_PRONOUN']
        elif self.filtering == 'he':
            filtered_df = df[(df['pronoun'] == 'he') | (df['pronoun'] == 'his') | (df['pronoun'] == 'him')]
        elif self.filtering == 'she':
            filtered_df = df[(df['pronoun'] == 'she') | (df['pronoun'] == 'her')]
        elif self.filtering == 'they':
            filtered_df = df[(df['pronoun'] == 'they') | (df['pronoun'] == 'them') | (df['pronoun'] == 'their')]
        elif self.filtering == 'xe':
            filtered_df = df[(df['pronoun'] == 'xe') | (df['pronoun'] == 'xem') | (df['pronoun'] == 'xyr')]
        elif self.filtering is not None:
            raise ValueError

        return filtered_df
    
    def get_tp(self):
        df = self._df.copy()
        df.loc[:, "found"] = df['verbalized_token'] == df['answer']
        if self.pairing != None:
            df_paired = df.groupby(self.pairing)["found"].min().reset_index().rename(columns={"found": "paired"})
            df = pd.merge(df, df_paired, on=self.pairing)
            return self._filter_df(df)["paired"].sum()
        else:
            return self._filter_df(df)["found"].sum()

    def get_tn(self):
        df = self._df.copy()
        df.loc[:, "found"] = df['verbalized_token'] != df['distractor']
        if self.pairing != None:
            df_paired = df.groupby(self.pairing)["found"].min().reset_index().rename(columns={"found": "paired"})
            df = pd.merge(df, df_paired, on=self.pairing)
            return self._filter_df(df)["paired"].sum()
        else:
            return self._filter_df(df)["found"].sum()

    def get_fn(self):
        df = self._df.copy()
        df.loc[:, "found"] = df['verbalized_token'] != df['answer']
        if self.pairing != None:
            df_paired = df.groupby(self.pairing)["found"].max().reset_index().rename(columns={"found": "paired"})
            df = pd.merge(df, df_paired, on=self.pairing)
            return self._filter_df(df)["paired"].sum()
        else:
            return self._filter_df(df)["found"].sum()

    def get_fp(self):
        df = self._df.copy()
        df.loc[:, "found"] = df['verbalized_token'] == df['distractor']
        if self.pairing != None:
            df_paired = df.groupby(self.pairing)["found"].max().reset_index().rename(columns={"found": "paired"})
            df = pd.merge(df, df_paired, on=self.pairing)
            return self._filter_df(df)["paired"].sum()
        else:
            return self._filter_df(df)["found"].sum()
    
    def get_precision(self):
        tp = self.get_tp()
        fp = self.get_fp()
        if tp+fp == 0:
            return 1.0
        return tp / (tp+fp)

    def get_recall(self):
        tp = self.get_tp()
        fn = self.get_fn()
        return tp / (tp+fn)

    def get_f1(self):
        precision = self.get_precision()
        recall = self.get_recall()
        f1 = (2*precision*recall)
        if f1 != 0:
            f1 = f1/(precision+recall)
        return f1
    
    def get_denom_accuracy(self):
        df = self._df.copy()
        df.loc[:, "found"] = df['verbalized_token'] == df['distractor']
        if self.pairing != None:
            df_paired = df.groupby(self.pairing)["found"].max().reset_index().rename(columns={"found": "paired"})
            df = pd.merge(df, df_paired, on=self.pairing)
        return len(self._filter_df(df))
    
    def get_accuracy(self):
        return self.get_tp() / self.get_denom_accuracy() # necessary because of how tn is calculated
    
    @property
    def filtering(self):
        return self._filtering
    
    @property
    def pairing(self):
        return self._pairing

    @filtering.setter
    def filtering(self, value):
        if not value in [None, 'nom', 'acc', 'poss', 'he', 'she', 'they', 'xe']:
            raise ValueError
        self._filtering = value

    @pairing.setter
    def pairing(self, value):
        if not value in [None, "template", "coref_dir"]:
            raise ValueError
        self._pairing = value


def main():
    flan_scaling = ['prompt_google_flan-t5-small',
                    'prompt_google_flan-t5-base',
                    'prompt_google_flan-t5-large',
                    'prompt_google_flan-t5-xl',
                    'prompt_google_flan-t5-xxl']

    spanberts = ['spanbert_base', 'spanbert_large']

    filtering = [None, 'nom', 'acc', 'poss', 'he', 'she', 'they', 'xe']

    pairing = sys.argv[1] if len(sys.argv) > 1 else None
  
    for source_data in ['double']: #, 'single', 'double_old']:
        print('results on', source_data)
        for system in flan_scaling + ['caw_coref', 'lingmess'] + spanberts:
            if os.path.exists(source_data + '/' + system + '.tsv'):
                print(f'{system}:')
                df = pd.read_csv(source_data + '/' + system + '.tsv', sep="\t")
                scorer = Scorer(df)
                scorer.pairing = pairing
                for filter_ in filtering:
                    print('filtering:', filter_)
                    scorer.filtering = filter_
                    print(f"{scorer.get_precision():2.2f} {scorer.get_recall():2.2f} {scorer.get_f1():2.2f} {scorer.get_accuracy():2.2f}")
                print()


if __name__ == '__main__':
    main()
