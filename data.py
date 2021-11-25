import os.path
from typing import Tuple, Union
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from glob import glob

import constants


def load_category_map(path):
    cat2code = {}
    with open(path, 'rt') as fin:
        for line in fin:
            category, code = line.split()
            cat2code[category] = int(code)
    code2cat = [cat for cat, code in sorted(cat2code.items(), key=lambda x:x[1])]
    return cat2code, code2cat


def load_raw_documents(path):
    """
    Reads a labelled colletion of documents.
    File fomart <0 or 1>\t<document>\n

    :param path: path to the labelled collection
    :return: a list of sentences, and a list of labels
    """
    all_sentences, all_labels = [], []
    file = open(path, 'rt', encoding='utf8').readlines()
    for line in file:
        line = line.strip()
        if line:
            try:
                label, sentence = line.split('\t')
                sentence = sentence.strip()
                label = int(label)
                if sentence:
                    all_sentences.append(sentence)
                    all_labels.append(label)
            except ValueError:
                print(f'format error in {line}')
    return all_sentences, all_labels


def load_raw_unlabelled_documents(path, vectorizer=None):
    with open(path, 'rt', encoding='utf-8') as file:
        documents = [d.strip() for d in file.readlines()]
    if vectorizer:
        documents = vectorizer.transform(documents)
    return documents, None


def load_vector_documents(path, nF=None):
    X, y = load_svmlight_file(path, n_features=nF, zero_based=True)
    y = y.astype(int)
    return X, y


def __gen_load_samples_with_groudtruth(path_dir:str, return_id:bool, ground_truth_path:str, load_fn, **load_kwargs):
    true_prevs = ResultSubmission.load(ground_truth_path)
    for id, prevalence in true_prevs.iterrows():
        sample, _ = load_fn(os.path.join(path_dir, f'{id}.txt'), **load_kwargs)
        yield (id, sample, prevalence) if return_id else (sample, prevalence)


def __gen_load_samples_without_groudtruth(path_dir:str, return_id:bool, load_fn, **load_kwargs):
    nsamples = len(glob(os.path.join(path_dir, '*.txt')))
    for id in range(nsamples):
        sample, _ = load_fn(os.path.join(path_dir, f'{id}.txt'), **load_kwargs)
        yield (id, sample) if return_id else sample


def gen_load_samples(path_dir:str, ground_truth_path:str = None, return_id=True, load_fn=load_vector_documents, **load_kwargs):
    if ground_truth_path is None:
        # the generator function returns tuples (docid:str, sample:csr_matrix or str)
        gen_fn = __gen_load_samples_without_groudtruth(path_dir, return_id, load_fn, **load_kwargs)
    else:
        # the generator function returns tuples (docid:str, sample:csr_matrix or str, prevalence:ndarray)
        gen_fn = __gen_load_samples_with_groudtruth(path_dir, return_id, ground_truth_path, load_fn, **load_kwargs)
    for r in gen_fn:
        yield r


class ResultSubmission:

    def __init__(self):
        self.df = None

    def __init_df(self, categories:int):
        if not isinstance(categories, int) or categories < 2:
            raise TypeError('wrong format for categories: an int (>=2) was expected')
        df = pd.DataFrame(columns=list(range(categories)))
        df.index.set_names('id', inplace=True)
        self.df = df

    @property
    def n_categories(self):
        return len(self.df.columns.values)

    def add(self, sample_id:int, prevalence_values:np.ndarray):
        if not isinstance(sample_id, int):
            raise TypeError(f'error: expected int for sample_sample, found {type(sample_id)}')
        if not isinstance(prevalence_values, np.ndarray):
            raise TypeError(f'error: expected np.ndarray for prevalence_values, found {type(prevalence_values)}')
        if self.df is None:
            self.__init_df(categories=len(prevalence_values))
        if sample_id in self.df.index.values:
            raise ValueError(f'error: prevalence values for "{sample_id}" already added')
        if prevalence_values.ndim != 1 and prevalence_values.size != self.n_categories:
            raise ValueError(f'error: wrong shape found for prevalence vector {prevalence_values}')
        if (prevalence_values<0).any() or (prevalence_values>1).any():
            raise ValueError(f'error: prevalence values out of range [0,1] for "{sample_id}"')
        if np.abs(prevalence_values.sum()-1) > constants.ERROR_TOL:
            raise ValueError(f'error: prevalence values do not sum up to one for "{sample_id}"'
                             f'(error tolerance {constants.ERROR_TOL})')

        self.df.loc[sample_id] = prevalence_values

    def __len__(self):
        return len(self.df)

    @classmethod
    def load(cls, path: str) -> 'ResultSubmission':
        df = ResultSubmission.check_file_format(path)
        r = ResultSubmission()
        r.df = df
        return r

    def dump(self, path:str):
        ResultSubmission.check_dataframe_format(self.df)
        self.df.to_csv(path)

    def prevalence(self, sample_id:int):
        sel = self.df.loc[sample_id]
        if sel.empty:
            return None
        else:
            return sel.values.flatten()

    def iterrows(self):
        for index, row in self.df.iterrows():
            prevalence = row.values.flatten()
            yield index, prevalence

    @classmethod
    def check_file_format(cls, path) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        df = pd.read_csv(path, index_col=0)
        return ResultSubmission.check_dataframe_format(df, path=path)

    @classmethod
    def check_dataframe_format(cls, df, path=None) -> Union[pd.DataFrame, Tuple[pd.DataFrame, str]]:
        hint_path = ''  # if given, show the data path in the error message
        if path is not None:
            hint_path = f' in {path}'

        if df.index.name != 'id' or len(df.columns) < 2:
            raise ValueError(f'wrong header{hint_path}, '
                             f'the format of the header should be "id,0,...,n-1", '
                             f'where n is the number of categories')
        if [int(ci) for ci in df.columns.values] != list(range(len(df.columns))):
            raise ValueError(f'wrong header{hint_path}, category ids should be 0,1,2,...,n-1, '
                             f'where n is the number of categories')
        if df.empty:
            raise ValueError(f'error{hint_path}: results file is empty')
        elif len(df) != constants.DEV_SAMPLES and len(df) != constants.TEST_SAMPLES:
            raise ValueError(f'wrong number of prevalence values found{hint_path}; '
                             f'expected {constants.DEV_SAMPLES} for development sets and '
                             f'{constants.TEST_SAMPLES} for test sets; found {len(df)}')

        ids = set(df.index.values)
        expected_ids = set(range(len(df)))
        if ids != expected_ids:
            missing = expected_ids - ids
            if missing:
                raise ValueError(f'there are {len(missing)} missing ids{hint_path}: {sorted(missing)}')
            unexpected = ids - expected_ids
            if unexpected:
                raise ValueError(f'there are {len(missing)} unexpected ids{hint_path}: {sorted(unexpected)}')

        for category_id in df.columns:
            if (df[category_id] < 0).any() or (df[category_id] > 1).any():
                raise ValueError(f'error{hint_path} column "{category_id}" contains values out of range [0,1]')

        prevs = df.values
        round_errors = np.abs(prevs.sum(axis=-1) - 1.) > constants.ERROR_TOL
        if round_errors.any():
            raise ValueError(f'warning: prevalence values in rows with id {np.where(round_errors)[0].tolist()} '
                              f'do not sum up to 1 (error tolerance {constants.ERROR_TOL}), '
                              f'probably due to some rounding errors.')

        return df












