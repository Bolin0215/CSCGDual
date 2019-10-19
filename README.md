## Dual Training for CS and CG

### Requirements
- Python 2.7.12
- Pytorch 0.3.1
- CUDA 8.0
- NLTK 3.3
- Java 1.8.0

### Train and Test

Our code is based on https://github.com/yistLin/pytorch-dual-learning.

- Preprocess the data and put the data in the ''data'' directory. 
    - Java dataset can be found at https://github.com/xing-hu/TL-CodeSum.
    - Python dataset can be found at https://github.com/wanyao1992/code_summarization_public. 
  
- Generating vocabulary for pretraining and dual training
```
    cd nmt
    python vocab.py --train_src ${source} --train_tgt ${target} --output ${output} --include_singleton
```
  
- Pretraining CS and CG models

The default setting in the example shell script is for pretraining CS model. If you want to pretrain CG model, you should modify the values of parameters L1 and L2. 
```
    sh pretrain.sh
```

- Pretraining language models
```
    cd lm
    python main.py --data ${path_to_data} --save ${path_to_save_model}
```

- Compute P(x) and P(y)
```
    python2 compute_lm_prob.py
```

- Train dual model
```
    sh train-dual.sh
```
- Test one model
```
    sh test.sh ${src} ${tgt} ${model} ${output} ${gpu}
```

### Evaluation
Evaluation code is based on https://github.com/tylin/coco-caption
```
    cd evaluation
    python2 evaluate.py
```