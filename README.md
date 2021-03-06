
---   
<div align="center">    
 
# Emotion Recognition on large video dataset based on Convolutional Feature Extractor(CNN) and Recurrent Neural Network(RNN)

[![Paper](http://img.shields.io/badge/paper-arxiv.2006.11168-B31B1B.svg)](https://arxiv.org/abs/2006.11168)
[![Challenge](http://img.shields.io/badge/ABAW-2020-4b44ce.svg)](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/)
[![Workshop](http://img.shields.io/badge/FG-2020-4b44ce.svg)](https://ibug.doc.ic.ac.uk/resources/affect-recognition-wild-unimulti-modal-analysis-va/) 
[![Conference](http://img.shields.io/badge/IPAS.IEEE-2020-B31B1B.svg)](https://ipas.ieee.tn)
</div>
 
## Description
This repository holds the Tensorflow Keras implementation of the approach described in our report [Emotion Recognition on large video dataset based on Convolutional Feature Extractor and Recurrent Neural Network](https://arxiv.org/abs/2006.11168), which is used for our entry to [ABAW Challenge 2020 (VA track)](https://ibug.doc.ic.ac.uk/resources/fg-2020-competition-affective-behavior-analysis/).

The main strength of this approach is separating training of CNN and RNN to feed RNN with simple feature vectors extracted from frames. 
It makes the RNN training process faster and gives the ability to exploit many video sequences with a lot of frames in a single batch. 
Featured extraction is done by simple CNN architecture proposed in [How Deep Neural Networks Can Improve Emotion Recognition on Video Data](https://arxiv.org/abs/1602.07377). 
Exploiting of temporal dynamics of video data are done by GRU.

We provide models trained on Aff-Wild2 in [data/checkpoints](data/checkpoints) and tensorboard logs during training this models in [data/logs](data/logs).
Also, all experiments such as validation of the CNN on small datasets, visualization of CNN layers, training and predicting are in [notebooks](notebooks).

## Getting Started   
### Prepare environment
Firstly, install dependencies
```bash
# clone project   
git clone https://github.com/DenisRang/Combined-CNN-RNN-for-emotion-recognition.git
python3 -m pip install -r requirements.txt --user
```
### Training
To train models, change paths in `config.py` with your ones. Then:
* train CNN model using `train_cnn.py` 
* change CNN extractor model path in `config.py` with new one
* extract features from cropped-aligned frames using `extract_features.py`
* train RNN model on created features from previous step using `train_rnn.py`
### Predicting
To predict emotions using our pretrained models, change paths in `config.py` with your ones and then run `predict.py`

### Other
There are more scripts for different purposes like additional analysis. 
Every script is provided with describing comments on the top.
## Citation   
```
@article{denis2020,
Title - {Emotion Recognition on large video dataset based on Convolutional Feature Extractor and Recurrent Neural Network},
Author {Denis Rangulov and Muhammad Fahim},
booktitle={IEEE International Conference on Image Processing, Applications and Systems},
 year={2020},
organization={IEEE}
}
```
