# Saliency-free and Aesthetic-aware Panoramic Video Navigation

Chenglizhao Chen, Guangxiao Ma\*, Wenfeng Song, Shuai Li, Aimin Hao, and Hong Qin (\*indicate equal contribution)


![](/demo/01.gif)

Paper: [ArXiv pre-print](), [Open access]()

# Prerequisites

- Linux
- NVIDIA GPU + CUDA 8.0 + CuDNNv5.1
- Python 2.7 with numpy
- [Tensorflow](https://www.tensorflow.org/) 1.2.1


## Getting Started
- Change the version you like:

  We provide both `0.12` and `1.2.1` version of Tensorflow implementation
You may choose the ideal version to use

- Clone this repo for formating the input data:

- Download our [dataset](#dataset) and [pre-trained model](#pre-trained-model)

After run the scripts you will see multiple links
```bash
python require.py
```
Please download our model and dataset and place it under `./checkpoint` and `./data`, respectively.


# Usage
To train a model with downloaded dataset:
```bash
python main.py --mode train --gpu 0 -d bmx -l 10 -b 16 -p classify --opt Adam
```
Then
```bash
python main.py --mode train --gpu 0 -d bmx -l 10 -b 16 -p regress --opt Adam --model checkpoint/bmx_16boxes_lam10.0/bmx_lam1_classify_best_model
```

To test with an existing model:
```bash
python main.py --mode test --gpu 0 -d bmx -l 10 -b 16 -p classify --model checkpoint/bmx_16boxes_lam10.0/bmx_lam1_classify_best_model
```
Or,
```bash
python main.py --mode test --gpu 0 -d bmx -l 10 -b 16 -p regress --model checkpoint/bmx_16boxes_lam10.0/bmx_lam10.0_regress_best_model
```

To get prediction with an existing model:
```bash
python main.py --mode pred --model checkpoint/bmx_16boxes_lam10.0/bmx_lam10.0_regress_best_model --gpu 0 -d bmx -l 10 -b 16 -p regress -n zZ6FlZRLvek_6
```

## Pre-trained Model
Please download the trained model for TensorFlow v1.2.1 [here]().
You can use `--model {model_path}` in `main.py` to load the model. 

## Dataset

### Pipeline testing
We provide a small testing clip-based datafile. Please download it [here](). And you can use this toy datafile to go though our data process pipeline.

### Testing on our *batch-based dataset* for accuracy and smoothness
If you want to reproduce the results on our dataset, please download the dataset [here](), label [here]() and place it under `./data`.

### Testing on our *clip-based dataset* for generating trajectories
Please download the *clip-based dataset* [here]()

# Cite
If you find our code useful for your research, please cite
```bibtex

```

# Author
mgx(#->@)jbeta.net
