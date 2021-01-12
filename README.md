# Neural Mask Generator: Learning to Generate the Adaptive Masking for Language Model Adaptation

## Anonymous Code for EMNLP Submission
We provide sample dataset and code for SQuAD v1.1.

The Code for Language model is based on https://github.com/huggingface/transformers.

Please download SQuAD.jsonl from mrqa.github.io and add it into dataset/SQuAD folder for SQuAD v1.1 demo.

## How to Run
1. Meta-training
```
./run_train.sh 2020xxxx qa squad bert $GPU 
```

2. Meta-testing
```
./run_test.sh 2020xxxx output/squad/bert/2020xxxx_neural qa squad bert $GPU
$GPUNUM 50 3 0.05
```

