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
2: 2 supporting facts | 0.1 | 3.0 | -
3: 3 supporting facts | 4.1 | ? | -
4: 2 argument relations | 0 | 0 | 0
5: 3 argument relations | 0.3 | ? | 1.5 
6: yes/no questions | 0.2 | 0 | 0
7: counting | 0 | 0 | -
8: lists/sets | 0.5 | 0 | 3.0 
9: simple negation | 0.1 | 0 | 0
10: indefinite knowledge | 0.6 | 0 | 0
11: basic coreference | 0.3 | 0 | 13.0
12: conjunction | 0 | 0 | 0.6
13: compound coreference | 1.3 | 0 | -
14: time reasoning | 0 | 0 | 3 
15: basic deduction | 0 | 0 | 0
16: basic induction | 0.2 | 0 | -
17: positional reasoning | 0.5 | 1.7 | 0.01
18: size reasoning | 0.3 | 1.5 | 7
19: path finding | 2.3 | 0 | -
20: agents motivation | 0 | 0 | 0
**Failed Tasks** | 0 | ?
**Mean Error** | 0.5 | ?

All tasks were run under the same settings. Not all tasks acheived optimal results under the same settings.

## Implementation-specific points that weren't clear to me from the paper:
1. Same parameters for in Prelu for memory and output model. Number of parameters equal to number of units.
2. Distinct positional masks for query and story. 
3. Positional mask size for max sentence size and not vocab size
4. CyclicLR: LR scheduling in the paper doens't seem to work for all tasks.
6. Pre-processing: Some models don't train in absence of puncts. 

