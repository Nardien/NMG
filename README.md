# Neural Mask Generator: Learning to Generate the Adaptive Maskings for Language Model Adaptation

This is the **Pytorch Implementation** for the paper _Neural Mask Generator: Learning to Generate the Adaptive Maskings for Language Model Adaptation_ (Accepted at EMNLP 2020, https://www.aclweb.org/anthology/2020.emnlp-main.493/)

Now the code supports the Question Answering task and Text Classification task.

## Abstract
<img align="right" width="250" src="https://github.com/Nardien/NMG/blob/master/images/concept_figure.png">
We propose a method to automatically generate a domain- and task-adaptive maskings of the given text for self-supervised pre-training, such that we can effectively adapt the language model to a particular target task (e.g. question answering). Specifically, we present a novel reinforcement learning-based framework which learns the masking policy, such that using the generated masks for further pre-training of the target language model helps improve task performance on unseen texts. We use off-policy actor-critic with entropy regularization and experience replay for reinforcement learning, and propose a Transformer-based policy network that can consider the relative importance of words in a given text. We validate our Neural Mask Generator (NMG) on several question answering and text classification datasets using BERT and DistilBERT as the language models, on which it outperforms rule-based masking strategies, by automatically learning optimal adaptive maskings.

## Prerequisites
- Python 3.6
- pytorch==1.4.0
- transformers==3.0.2
- tqdm

## Dataset
For Question Answering (QA) dataset, please refer to https://github.com/mrqa/MRQA-Shared-Task-2019 for NewsQA and https://github.com/panushri25/emrQA for emrQA.

For all QA dataset, we preprocess them into the MRQA format.

For Text Classification dataset, please refer to https://github.com/allenai/dont-stop-pretraining.

As the text corpus, you should extract context from each dataset and build it as the distinct dataset. (We will provide it as the downloadable link if possible.)

## How to Run
1. Meta-training

Question Answering
```
./run_train.sh 2020xxxx qa squad bert $GPU 
```
Text Classification
```
./run_train.sh 2020xxxx glue chemprot bert $GPU
```

2. Meta-testing

Question Answering
```
./run_test.sh 2020xxxx output/squad/bert/2020xxxx_neural qa squad bert $GPU
$GPUNUM 50 3 0.05
```
Text Classification
```
./run_test.sh 2020xxxx output/chemprot/bert/2020xxxx_neural glue chemprot bert $GPU
$GPUNUM 50 3 0.05
```

I have checked that the code successfully works for both tasks, however, please feel free to leave it on Issues if you find any problems.

## TODOs

- [x]  Update the code for support on the text classification
- [x]  Clean up the code including the removal of useless configurations
- [x]  Add the text corpus extraction code.

## Citation
If you found this work useful, please cite our work.
```
@inproceedings{DBLP:conf/emnlp/KangHH20,
  author    = {Minki Kang and
               Moonsu Han and
               Sung Ju Hwang},
  title     = {Neural Mask Generator: Learning to Generate Adaptive Word Maskings
               for Language Model Adaptation},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural
               Language Processing, {EMNLP} 2020, Online, November 16-20, 2020},
  pages     = {6102--6120},
  year      = {2020},
}
```
