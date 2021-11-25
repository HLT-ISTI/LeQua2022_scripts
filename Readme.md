# LeQua 2022: Learning to Quantify

This repository contains the official [evaluation script](evaluate.py) that will be used
for evaluating submissions in the LeQua2022 (the 1st edition of the 
CLEF “Learning to Quantify” lab) competition. 
This repository also provides a [format checker](format_checker.py),
in order to allow participants to check that the format of their submissions
is correct.
Additionally, some helper methods are made available for the conveniency of participants.

## What is LeQua 2022?
LeQua2022 is the 1st edition of the CLEF “Learning to Quantify” lab.
The aim of this competitions is to allow the comparative evaluation 
of methods for “learning to quantify” in textual datasets, i.e., methods
for training predictors of the relative frequencies of the 
classes of interest in sets of unlabelled textual documents.
For further details, please visit [the official LeQua2022's site](https://lequa2022.github.io/).

## The evaluation script

The [evaluation script](evaluate.py) takes two results files, one for
the true prevalence values (ground truth) and another for the estimated prevalence
values (a submission file), and computes the error (in terms of `mean absolute error` and
`mean relative absolute error`). The script can be run from command line as follows (use
`--help` to display the usage information):

```
usage: evaluate.py [-h] [--output SCORES-PATH]
                   TASK TRUE-PREV-PATH PRED-PREV-PATH

LeQua2022 official evaluation script

positional arguments:
  TASK                  Task name (T1A, T1B, T2A, T2B)
  TRUE-PREV-PATH        Path of ground truth prevalence values file (.csv)
  PRED-PREV-PATH        Path of predicted prevalence values file (.csv)

optional arguments:
  -h, --help            show this help message and exit
  --output SCORES-PATH  Path where to store the evaluation scores
```

The errors are displayed in standard output and, optionally, dumped on a txt file.
For example:

> python3 evaluate.py T1A ./data/T1A/public/dev_prevalences.csv ./my_submission/estimated_prevalences.csv --output scores.txt

*Note* that the first file corresponds to the ground truth prevalence values, and the second file
corresponds to the estimated ones. The order is not interchangeable since 
relative absolute error is not symmetric.

## The format checker

The [format checker](format_checker.py) serves the purpose of guaranteeing
participants that their submission files contain no errors.
See the usage information (by typing `--help`):

```
usage: format_checker.py [-h] PREVALENCEFILE-PATH

LeQua2022 official format-checker script

positional arguments:
  PREVALENCEFILE-PATH  Path to the file containing prevalence values to check

optional arguments:
  -h, --help           show this help message and exit
```

Some mock submission files are provided as examples. For example, running:

> python3 format_checker.py ./data/T1A/public/dummy_dev_predictions.T1A.csv

Will produce the output:

> Format check: passed

If the format is not correct, the check will not pass, and the checker will
display some hint regarding the type of error encountered.

## The submission files format:

Submission files will consist of `csv` (comma separated values) files.
The format should comply with the following conditions:
* The header is: `id, 0, 1, <...>, n-1` where `n` is the number of classes 
  (`n=2` in tasks T1A and T1B, and `n=28` in tasks T2A and T2B)
* Every row consists of a document id (an integer) followed by the
  class prevalence values for each class. E.g., a valid row could be 
  `5,0.7,0.3`, meaning that the sample with id 5 (i.e., the file `5.txt`) has prevalence values of 0.7 for the class
  0 and 0.3 for the class 1.
* There should be exactly 1000 rows (for tasks T1A and T2A) or exactly 5000 rows 
  (for tasks T1B and T2B), after the header.
* Document ids must not contain gaps, and should start by 0.
* Prevalence values should be in the range [0,1] for all classes, and sum
  up to 1 (with an error tolerance of 1E-3).

The easiest way to create a valid submission file is by means of the
class `ResultSubmission` defined in [data.py](data.py). See the following 
section in which some useful functions are discussed.

## Utilities

Some useful functions for managing the data are made available in
[data.py](data.py) for your convenience. The following example illustrates
some of them, and shows how to iterate over data samples both for the
vector tasks (T1A and T1B -- illustrated for T1A) and for the raw text tasks
(T2A and T2B -- illustrated for T2B).

Before getting started, let's simulate the outputs of any quantifier 
by means of some mock predictions: 

```
from data import *
import numpy as np

def mock_prediction(n_classes):
    predicted_prevalence = np.random.rand(n_classes)
    predicted_prevalence /= predicted_prevalence.sum()
    return predicted_prevalence
```

This first example shows how to create a valid submission file for task T1A.
Let us assume we already have a trained quantifier, and we want to assess
its performance on the development set (for which the true prevalence
values are known). 
Assume the development samples are located in `./data/T1A/public/dev_vectors`  
and the true prevalence are in `./data/T1A/public/dev_prevalences.csv`.
This could be carried out as follows:

```
# create a new submission file
submission_T1A = ResultSubmission()

path_dir = './data/T1A/public/dev_vectors'
ground_truth_path = './data/T1A/public/dev_prevalences.csv'

# iterate over devel samples; the task is T1A and so the samples are in vector form;
# the function that loads samples in vector form is "load_vector_documents"
for id, sample, prev in gen_load_samples(path_dir, ground_truth_path, return_id=True, load_fn=load_vector_documents):
    # replace this with the predictions generated by your quantifier
    predicted_prevalence = mock_prediction(n_classes=2)
    
    # append the predictions to the submission file
    submission_T1A.add(id, predicted_prevalence)
    
# dump to file
submission_T1A.dump('mock_submission.T1A.csv')
```

Participants can now use the [evaluation script](evaluate.py) to 
evaluate the quality of their predictions with respect to the ground truth.

After the test set is released (see the [timeline](https://lequa2022.github.io/timeline/)
for further details), participants will be asked
to generate predictions for the test samples, and to submit a prediction
file. Note that the ground truth prevalence values of the test samples
will not be made available until the competition finishes. 
The following script illustrates how to iterate over test examples
and generate a valid submission file.

```
submission_T1A = ResultSubmission()
path_dir = './data/T1A/future/test_vectors'

# the iterator has no access to the true prevalence values, since a
# file containing the ground truth values is not indiciated 
for id, sample in gen_load_samples(path_dir, load_fn=load_vector_documents):
    submission_T1A.add(id, mock_prediction(n_classes=2))
submission_T1A.dump('mock_submission.T1A.csv')
```

The only difference concerning tasks T1B and T2B regards the data loader function.
The function `load_raw_unlabelled_documents` implements this process; see, e.g.:

```
submission_T2B = ResultSubmission()
path_dir = './data/T2B/public/dev_documents'
ground_truth_path = './data/T2B/public/dev_prevalences.csv'
for id, sample, prev in gen_load_samples(path_dir, ground_truth_path, load_fn=load_raw_unlabelled_documents):
    predicted_prevalence = mock_prediction(n_classes=28)
    submission_T2B.add(id, predicted_prevalence)
submission_T2B.dump('mock_submission.T2B.csv')
```

The following functions might be useful as well (implemented in [data.py](data.py)):
* load_vector_documents(path, nF=None): loads documents for tasks T1A and T1B. Note that
  only training documents are labelled. Development samples are (and test samples will be)
  unlabelled, although the format is the same (the label takes value -1 in all such cases)
* load_raw_documents(path): loads labelled documents for tasks T2A and T2B
