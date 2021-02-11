# Load packages

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from transformers import AutoConfig, AutoTokenizer, TFAutoModel

import seaborn as sns
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Adam

# Load the Datasets
df_train = pd.read_csv('/content/train.csv')
df_validation = pd.read_csv('/content/valid.csv')
df_test = pd.read_csv('/content/test.csv')

# initiate ParsBERT Tokenizer
tokenizer = AutoTokenizer.from_pretrained("HooshvareLab/bert-fa-base-uncased")


def encode_dataset(tokenizer, text_sequences, max_length):
    """Encodes each Sentence with BERT tokenizer and encoder

    Args:
        tokenizer: ParsBERT Tokenizer
        text_sequences: Input Sentences
        max_length: Maximum length needed for zero padding

    Returns:
        dictionary: encoded sentences and their corresponding attention masks
    """

    token_ids = np.zeros(shape=(len(text_sequences), max_length),
                         dtype=np.int32)

    for i, text_sequence in enumerate(text_sequences):
        encoded = tokenizer.encode(text_sequence)
        token_ids[i, 0:len(encoded)] = encoded

    attention_masks = (token_ids != 0).astype(np.int32)

    return {'input_ids': token_ids, 'attention_masks': attention_masks}

# Encode Datasets
encoded_train = encode_dataset(tokenizer, df_train['words'], 25)
encoded_validation = encode_dataset(tokenizer, df_validation['words'], 25)
encoded_test = encode_dataset(tokenizer, df_test['words'], 25)

# Creating Intent Map
intent_names = Path('/content/intent.txt').read_text('utf-8').split()
intent_map = dict((label, idx) for idx, label in enumerate(intent_names))

# Converting Intent labels to numbers using intent map
intent_train = df_train['intent_label'].map(intent_map).values
intent_validation = df_validation['intent_label'].map(intent_map).values
intent_test = df_test['intent_label'].map(intent_map).values

# creating Slot map
slot_names = ["[PAD]"]
slot_names += Path('/content/slots.txt').read_text('utf-8').strip().splitlines()
slot_map = {}
for label in slot_names:
    slot_map[label] = len(slot_map)
    
    
def encode_token_labels(text_sequences, slot_names, tokenizer, slot_map, max_length):
    """After tokenization, some slot lables should be extended, before encoding. this function takes care of both extending and encoding.

    Args:
        text_sequences: a list of sentences
        slot_names: a list of slot names
        tokenizer: ParsBERT Tokenizer
        slot_map: a dictionary used for encoding slot names
        max_length: Maximum length needed for zero padding

    Returns:
        array: an array of encoded slot names
    """

    encoded = np.zeros(shape=(len(text_sequences), max_length), dtype=np.int32)
    for i, (text_sequence, word_labels) in enumerate(zip(text_sequences, slot_names)):
        encoded_labels = []
        for word, word_label in zip(text_sequence.split(), word_labels.split()):
            tokens = tokenizer.tokenize(word)
            encoded_labels.append(slot_map[word_label])
            expand_label = word_label.replace("B-", "I-")
            if not expand_label in slot_map:
                expand_label = word_label
            encoded_labels.extend([slot_map[expand_label]] * (len(tokens) - 1))
        encoded[i, 1:len(encoded_labels) + 1] = encoded_labels
    return encoded

# encoding slot names
slot_train = encode_token_labels(df_train['words'], df_train['words_label'], tokenizer, slot_map, 25)
slot_validation = encode_token_labels(df_validation['words'], df_validation['words_label'], tokenizer, slot_map, 25)
slot_test = encode_token_labels(df_test['words'], df_test['words_label'], tokenizer, slot_map, 25)

# Introducing ParsBERT model
base_bert_model = TFAutoModel.from_pretrained("HooshvareLab/bert-fa-base-uncased")

# Define JointIntentAndSlotFilling model

class JointIntentAndSlotFillingModel(tf.keras.Model):
    def __init__(self, intent_num_labels=None, slot_num_labels=None,
                 model_name="HooshvareLab/bert-fa-base-uncased"):

        super().__init__(name="joint_intent_slot")

        self.bert = TFAutoModel.from_pretrained(model_name)
        self.intent_dropout = Dropout(0.3)
        self.slot_dropout = Dropout(0.7)
        self.intent_classifier = Dense(intent_num_labels)
        self.slot_classifier = Dense(slot_num_labels)



    def call(self, inputs, **kwargs):

      #bert
      sequence_output, pooled_output = self.bert(inputs, **kwargs).values()

      #slot classifier
      sequence_output = self.slot_dropout(sequence_output)
      slot_logits = self.slot_classifier(sequence_output)

      #intent classifier
      pooled_output = self.intent_dropout(pooled_output)
      intent_logits = self.intent_classifier(pooled_output)

      return slot_logits, intent_logits

# creating the model
joint_model = JointIntentAndSlotFillingModel(intent_num_labels=len(intent_map), slot_num_labels=len(slot_map))

# Define one classification loss for each output:
opt = Adam(learning_rate=3e-5, epsilon=1e-08)
losses = [SparseCategoricalCrossentropy(from_logits=True),
          SparseCategoricalCrossentropy(from_logits=True)]
metrics = [SparseCategoricalAccuracy('accuracy')]

joint_model.compile(optimizer=opt, loss=losses, metrics=metrics)

# training the model
history = joint_model.fit(
    encoded_train, (slot_train, intent_train),
    validation_data=(encoded_validation, (slot_validation, intent_validation)),
    epochs=4, batch_size=32)

true_intents = []
predicted_intents = []

# testing the model
for i in range(df_test.shape[0]):
  text = df_test.iloc[i]['words']
  intent = df_test.iloc[i]['intent_label']
  true_intents.append(intent_map[intent])
  inputs = tf.constant(tokenizer.encode(text))[None, :]  # batch_size = 1
  outputs = joint_model(inputs)
  slot_logits, intent_logits = outputs
  slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
  intent_id = intent_logits.numpy().argmax(axis=-1)[0]
  predicted_intents.append(intent_id)
cf_matrix = confusion_matrix(true_intents, predicted_intents)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, fmt='.2%', cmap='Blues')

# Following function predicts for a single sentence using our model
def show_predictions(text, tokenizer, model, intent_names, slot_names):
    """this function predicts for a single sentence using our model

    Args:
        text: a persian sentence
        tokenizer: ParsBERT tokenizer
        model: a Joint BERT intent classifier and slot filler
        intent_names: a dictionary of intent names with their corresponding IDs
        slot_names: a dictionary of slot names with their corresponding IDs
    """
    inputs = tf.constant(tokenizer.encode(text))[None, :] 
    outputs = model(inputs)
    slot_logits, intent_logits = outputs
    slot_ids = slot_logits.numpy().argmax(axis=-1)[0, 1:-1]
    intent_id = intent_logits.numpy().argmax(axis=-1)[0]
    print("## Intent:", intent_names[intent_id])
    print("## Slots:")
    for token, slot_id in zip(tokenizer.tokenize(text), slot_ids):
        print(f"{token:>10} : {slot_names[slot_id]}")

# here we print the output of model for 10 random sentences from test dataset
n = int(df_test.shape[0]/10) - 1
for i in range(10):
  text = df_test.iloc[i*n]['words']
  print(">>>>",text)
  show_predictions(text, tokenizer, joint_model, intent_names, slot_names)
