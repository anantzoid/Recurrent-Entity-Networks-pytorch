# Recurrent Entity Networks

Pytorch implementation of REN as described in [Tracking the World State with Recurrent Entity Networks](https://arxiv.org/abs/1612.03969). The code is adapted from the Tensorflow version by [Jim Fleming](https://github.com/jimfleming).  

Differences I noticed in the description of the model in the paper and how it was implemented in TF version (which ultimately worked)
- Parameteric ReLU has different learnable parameters for each piece-wise unit.
- Parameteric ReLU activation paramaters area shared between the memory module and output module.
- The hidden state of the RNN is initialized from the key vectors.

## Results

Percent error for each BABI task, comparing those in the paper, the available TF version, and the Pytorch version in this repo. 

Task | EntNet (paper) | EntNet (TF) | EntNet (Pytorch)
--- | --- | ---
1: 1 supporting fact | 0 | 0 | 0
2: 2 supporting facts | 0.1 | 3.0 | 1.3
3: 3 supporting facts | 4.1 | ? | 1.6
4: 2 argument relations | 0 | 0 | 0
5: 3 argument relations | 0.3 | ?
6: yes/no questions | 0.2 | 0
7: counting | 0 | 0
8: lists/sets | 0.5 | 0
9: simple negation | 0.1 | 0
10: indefinite knowledge | 0.6 | 0
11: basic coreference | 0.3 | 0
12: conjunction | 0 | 0
13: compound coreference | 1.3 | 0
14: time reasoning | 0 | 0
15: basic deduction | 0 | 0
16: basic induction | 0.2 | 0
17: positional reasoning | 0.5 | 1.7
18: size reasoning | 0.3 | 1.5
19: path finding | 2.3 | 0
20: agents motivation | 0 | 0
**Failed Tasks** | 0 | ?
**Mean Error** | 0.5 | ?


