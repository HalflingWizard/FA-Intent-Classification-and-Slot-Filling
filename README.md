# FA-Intent-Classification-and-Slot-Filling
A Joint Model for Intent Classification and Slot Filling in Persian Language using BERT

<p align="center">
  <img src="https://raw.githubusercontent.com/HalflingWizard/FA-Intent-Classification-and-Slot-Filling/main/img/header.png" />
</p>

This model is based on [This Paper](https://arxiv.org/abs/1902.10909) by Qian Chen, Zhu Zhuo and Wen Wang.
I created a small Persian Dataset and used [ParsBERT](https://arxiv.org/abs/2005.12515) (which is is pre-trained on large Persian corpora with various writing styles from numerous subjects) instead of BERT.
## Dataset
First, I wrote 600 questions in 4 different categories that a user might ask their voice assistant to do. these questions are publicly available [here](https://arxiv.org/abs/2005.12515). Then I used the EDA method (as described [here](https://www.kaggle.com/halflingwizard/persian-text-augmentation)) for data augmentation for train and validation sets only.
## Model
I created this model based on [this](https://www.kaggle.com/stevengolo/join-intent-classification-and-slot-filling) notebook in Kaggle.
