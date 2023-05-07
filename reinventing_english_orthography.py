# %% [markdown]
# # Reinventing English Orthography with GANs
# 
# ## Test baseline performance of current spelling
# 
# ### Create training and test data
# 
# This is for the baseline test of how well a Seq2Seq model can create the mapping between pronunciation and spelling. 
# 
# The model will be trained on 8,000 words and tested on another 2,000.

# %%
import random
import nltk
from nltk.corpus import cmudict
import numpy as np

random.seed(98405)
np.random.seed(98405)

# Download the CMU Pronouncing Dictionary
nltk.download('cmudict')

# Load the CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Choose a random sample of words from the dictionary
num_samples = 10000  # Choose the number of random words you want to select
random_words = random.sample(list(cmu_dict.keys()), num_samples)

# Create a new dictionary with the selected words
# Key: Spelling, Value: First pronunciation
random_word_pronunciations = {word: cmu_dict[word][0] for word in random_words}

# %% [markdown]
# ### Process the data to prepare for ML

# %%
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

tf.random.set_seed(98405)

# Split the data into training, validation, and testing sets
train_pairs, test_pairs = train_test_split(list(random_word_pronunciations.items()), test_size=0.2, random_state=42)
train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.125, random_state=42)

# Tokenize characters and phonemes
input_characters = sorted(set(char for word, _ in list(random_word_pronunciations.items()) for char in word))
target_phonemes = sorted(set(phoneme for _, pronunciation in list(random_word_pronunciations.items()) for phoneme in pronunciation))
target_phonemes = ['\t', '\n'] + target_phonemes

input_char_to_idx = {char: idx for idx, char in enumerate(input_characters)}
target_phoneme_to_idx = {phoneme: idx for idx, phoneme in enumerate(target_phonemes)}

# Prepare the input and target data
def encode_sequences(pairs, input_char_to_idx, target_phoneme_to_idx):
    input_data = [[input_char_to_idx[char] for char in word] for word, _ in pairs]
    target_data = [[target_phoneme_to_idx["\t"]] + [target_phoneme_to_idx[phoneme] for phoneme in pronunciation] + [target_phoneme_to_idx["\n"]] for _, pronunciation in pairs]
    return input_data, target_data

encoder_input_data, decoder_target_data = encode_sequences(train_pairs, input_char_to_idx, target_phoneme_to_idx)
val_encoder_input_data, val_decoder_target_data = encode_sequences(val_pairs, input_char_to_idx, target_phoneme_to_idx)
test_encoder_input_data, test_decoder_target_data = encode_sequences(test_pairs, input_char_to_idx, target_phoneme_to_idx)

# Pad the sequences to the same length
max_encoder_seq_length = max(len(seq) for seq in encoder_input_data)
max_decoder_seq_length = max(len(seq) for seq in decoder_target_data)

encoder_input_data = pad_sequences(encoder_input_data, maxlen=max_encoder_seq_length, padding='post')
decoder_target_data = pad_sequences(decoder_target_data, maxlen=max_decoder_seq_length, padding='post')

val_encoder_input_data = pad_sequences(val_encoder_input_data, maxlen=max_encoder_seq_length, padding='post')
val_decoder_target_data = pad_sequences(val_decoder_target_data, maxlen=max_decoder_seq_length, padding='post')

test_encoder_input_data = pad_sequences(test_encoder_input_data, maxlen=max_encoder_seq_length, padding='post')
test_decoder_target_data = pad_sequences(test_decoder_target_data, maxlen=max_decoder_seq_length, padding='post')

# Prepare the decoder input data (shifted right by one timestep)
decoder_input_data = np.zeros_like(decoder_target_data)
decoder_input_data[:, 1:] = decoder_target_data[:, :-1]

val_decoder_input_data = np.zeros_like(val_decoder_target_data)
val_decoder_input_data[:, 1:] = val_decoder_target_data[:, :-1]

# Convert target data to one-hot encoding
decoder_target_data = to_categorical(decoder_target_data, num_classes=len(target_phonemes))
val_decoder_target_data = to_categorical(val_decoder_target_data, num_classes=len(target_phonemes))

# reshape for use in sparse categorical crossentropy
#decoder_target_data = np.argmax(decoder_target_data, axis=-1)[:, :, np.newaxis]
#val_decoder_target_data = np.argmax(val_decoder_target_data, axis=-1)[:, :, np.newaxis]
#decoder_target_data = np.squeeze(decoder_target_data, axis=-1)
#val_decoder_target_data = np.squeeze(val_decoder_target_data, axis=-1)

# %% [markdown]
# ### Creating the model

# %%
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Attention, Concatenate, Bidirectional
from tensorflow.keras.models import Model

# Hyperparameters
batch_size = 64
epochs = 50
latent_dim = 256
num_encoder_tokens = len(input_characters)  # Unique input characters
num_decoder_tokens = len(target_phonemes)  # Unique target characters

# Encoder
encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder_lstm = Bidirectional(LSTM(latent_dim, return_sequences=True, return_state=True))
encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(enc_emb)
state_h = Concatenate()([forward_h, backward_h])
state_c = Concatenate()([forward_c, backward_c])
# Decoder
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(num_decoder_tokens, latent_dim)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim*2, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Attention mechanism
attention_layer = Attention()
attention_output = attention_layer([decoder_outputs, encoder_outputs])
decoder_concat_input = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs, attention_output])

# Dense layer
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_concat_input)

# Create the model
model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

# %% [markdown]
# ### Training the model

# %%
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])
model.fit([encoder_input_data[:, :, np.newaxis], decoder_input_data[:, :, np.newaxis]], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# %% [markdown]
# ### Model inference and evaluation

# %%
# Create encoder model
encoder_model = Model(encoder_inputs, [encoder_outputs, state_h, state_c])

# Decoder input states
decoder_state_input_h = Input(shape=(latent_dim*2,))
decoder_state_input_c = Input(shape=(latent_dim*2,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Decoder input
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = dec_emb_layer(decoder_inputs_single)

# Decoder LSTM
decoder_outputs_single, state_h_single, state_c_single = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)
decoder_states_single = [state_h_single, state_c_single]

# Attention mechanism
attention_output_single = attention_layer([decoder_outputs_single, encoder_outputs])
decoder_concat_input_single = tf.keras.layers.Concatenate(axis=-1)([decoder_outputs_single, attention_output_single])

# Dense layer
decoder_outputs_single = decoder_dense(decoder_concat_input_single)

# Decoder model
decoder_model = Model(
    [decoder_inputs_single, encoder_outputs] + decoder_states_inputs,
    [decoder_outputs_single] + decoder_states_single
)

def decode_sequence(input_seq):
    # Encode the input as state vectors
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate an empty target sequence of length 1
    target_seq = np.zeros((1, 1))

    # Initialize the first character of the target sequence as the start token
    target_seq[0, 0] = target_phoneme_to_idx["\t"]

    # Create a stop condition flag
    stop_condition = False

    # Initialize the decoded sequence
    decoded_phonemes = []

    # Loop until the stop condition is met
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq, e_out] + [e_h, e_c])

        # Sample a token (phoneme)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_phoneme = target_phonemes[sampled_token_index]

        # Append the phoneme to the decoded sequence
        decoded_phonemes.append(sampled_phoneme)

        # Check for the stop condition: end of sequence token or maximum length reached
        if sampled_phoneme == "\n" or len(decoded_phonemes) > max_decoder_seq_length:
            stop_condition = True

        # Update the target sequence (length 1)
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index

        # Update the states
        e_h, e_c = h, c

    return decoded_phonemes


# %% [markdown]
# ### Output model results

# %%
input_word = "kitten"
input_sequence = np.array([input_char_to_idx[char] for char in input_word])
input_sequence = pad_sequences([input_sequence], maxlen=max_encoder_seq_length, padding='post')
predicted_phonemes = decode_sequence(input_sequence)
print(f"Predicted pronunciation for '{input_word}': {predicted_phonemes}")

# %%
import csv

# Define the CSV file name
csv_file_name = "spelling_phoneme_predictions.csv"

# Open the CSV file in write mode
with open(csv_file_name, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write the header row
    csv_writer.writerow(["Input Spelling", "True Phoneme Sequence", "Predicted Phoneme Sequence"])

    # Iterate through the input phoneme sequences
    for seq_index in range(len(test_pairs)):
        
        input_word = test_pairs[seq_index][0]
        input_sequence = np.array([input_char_to_idx[char] for char in input_word])
        input_sequence = pad_sequences([input_sequence], maxlen=max_encoder_seq_length, padding='post')

        # Write the input phoneme sequence, true spelling, and predicted spelling as a new row in the CSV
        csv_writer.writerow([test_pairs[seq_index][0], test_pairs[seq_index][1], decode_sequence(input_sequence)[1:-1]])


# %% [markdown]
# ## Recreating English orthography with transformers
# 
# ### Data prep

# %%
# Create data from random word pronunciations

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# split spelling and phonemes
all_spelling = list(random_word_pronunciations.keys())
all_phonemes = list(random_word_pronunciations.values())

# get maximum sequence length
max_seq_len = 22

# %%
# Prepare your data:
# Since you've already integer coded and padded the words and their pronunciations, you just need to split the dataset into training and validation sets. 
# This will allow you to train your model on a portion of the data and evaluate its performance on unseen data.

import re

# split spelling and phonemes
all_spelling = list(random_word_pronunciations.keys())
all_phonemes = list(random_word_pronunciations.values())

# split spelling and phonemes
spelling_sequences = list(random_word_pronunciations.keys())
phoneme_sequences = list(random_word_pronunciations.values())

unwanted_chars = set(['"', "'", '-', '.', '_', '{'])

def remove_chars(word, chars_to_remove):
    return ''.join([char for char in word if char not in chars_to_remove])

def clean_text(text, chars_to_remove):
    pattern = '[' + re.escape(''.join(chars_to_remove)) + ']'
    return re.sub(pattern, '', text)

# Remove unwanted characters from spelling sequences
cleaned_spelling_sequences = [remove_chars(word, unwanted_chars) for word in spelling_sequences]

# Create dictionaries for encoding and decoding
char_to_id = {char: idx for idx, char in enumerate(sorted(set(''.join(cleaned_spelling_sequences))), 3)}
phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(sorted(set([p for word in phoneme_sequences for p in word])), 3)}

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

max_length = 26

special_tokens = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]

def train_unigram_tokenizer(texts, vocab_size=1000):
    tokenizer = Tokenizer(models.Unigram())
    trainer = trainers.UnigramTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    return tokenizer

# Function to tokenize and pad sequences with BPE
def tokenize_and_pad_bpe(tokenizer, sequences, max_length, start_token='<s>', end_token='</s>'):
    tokenized_sequences = []
    for seq in sequences:
        tokens = [start_token] + tokenizer.encode(seq).tokens + [end_token]
        padded_tokens = tokens + ['<pad>'] * (max_length - len(tokens))
        tokenized_sequences.append(padded_tokens[:max_length])
    return tokenized_sequences

unwanted_chars = ['"', "'", '-', '.', '_', '{', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cleaned_spellings = [clean_text(word, unwanted_chars) for word in spelling_sequences]

# Pre-process spellings and phonemes
cleaned_spellings = [''.join(list(word)) for word in cleaned_spellings]

# Train a unigram tokenizer for spellings
spelling_tokenizer = train_unigram_tokenizer(cleaned_spellings, vocab_size=100)

tokenized_spellings = tokenize_and_pad_bpe(spelling_tokenizer, cleaned_spellings, max_length)

import string

def separate_numbers_from_phonemes(sequences):
    separated_sequences = []
    for seq in sequences:
        separated_seq = []
        for phoneme in seq:
            split_phoneme = re.findall(r'[A-Za-z]+|\d+', phoneme)
            for p in split_phoneme:
                if p != '0':
                    separated_seq.append(p)
        separated_sequences.append(separated_seq)
    return separated_sequences

# Separate numbers from phonemes in phoneme sequences
phoneme_sequences = separate_numbers_from_phonemes(phoneme_sequences)

# Create a mapping between phonemes and unique symbols
unique_phonemes = set([phoneme for seq in phoneme_sequences for phoneme in seq])
symbols = iter(string.ascii_uppercase + string.ascii_lowercase + string.digits + string.punctuation)
phoneme_to_symbol = {phoneme: next(symbols) for phoneme in unique_phonemes}
symbol_to_phoneme = {v: k for k, v in phoneme_to_symbol.items()}

# Function to substitute phonemes with symbols
def substitute_phonemes_with_symbols(sequences, mapping):
    substituted_sequences = []
    for seq in sequences:
        substituted_seq = [mapping[phoneme] for phoneme in seq]
        substituted_sequences.append(substituted_seq)
    return substituted_sequences

# Substitute phonemes with symbols in phoneme sequences
symbol_substituted_phonemes = substitute_phonemes_with_symbols(phoneme_sequences, phoneme_to_symbol)

# Train a unigram tokenizer for phonemes with symbols
symbol_phoneme_texts = [''.join(p) for p in symbol_substituted_phonemes]
symbol_phoneme_tokenizer = train_unigram_tokenizer(symbol_phoneme_texts, vocab_size=85)

# Tokenize the symbol-substituted phoneme sequences without special tokens
tokenized_symbol_phonemes = tokenize_and_pad_bpe(symbol_phoneme_tokenizer, symbol_phoneme_texts, max_length)

def replace_symbols_with_phonemes(tokenized_sequences, mapping):
    replaced_sequences = []
    for seq in tokenized_sequences:
        replaced_seq = []
        for token in seq:
            if token in mapping:
                replaced_token = mapping[token]
            elif token in special_tokens:
                replaced_token = token
            else:
                replaced_char = []
                for char in token:
                    replaced_char.append(mapping[char])
                replaced_token = '-'.join(replaced_char)
            replaced_seq.append(replaced_token)
        replaced_sequences.append(replaced_seq)
    return replaced_sequences

tokenized_phonemes = replace_symbols_with_phonemes(tokenized_symbol_phonemes, symbol_to_phoneme)

# Tokenize characters and phonemes
target_phonemes = sorted(set(phoneme for pronunciation in tokenized_phonemes for phoneme in pronunciation))

phoneme_to_id = {phoneme: idx for idx, phoneme in enumerate(target_phonemes)}

# Convert phonemes to integers
def encode_seqs(sequences, max_seq_len, start_char='\t', end_char='\n'):
    # Append start and end characters to each sequence
    sequences = [seq for seq in sequences]

    label_encoder = LabelEncoder()
    all_phonemes = [phoneme for seq in sequences for phoneme in seq]
    label_encoder.fit(all_phonemes)
    integer_sequences = [label_encoder.transform(seq) for seq in sequences]
    return integer_sequences

# Use encoding on phoneme data
tokenized_phonemes_int = encode_seqs(tokenized_phonemes, max_seq_len)
train_phonemes, test_phonemes = train_test_split(tokenized_phonemes_int, test_size=0.2, random_state=42)

input_characters = sorted(set(char for word in tokenized_spellings for char in word))
char_to_id = {char: idx for idx, char in enumerate(input_characters)}

tokenized_spelling_int = encode_seqs(tokenized_spellings, max_seq_len)
train_spellings, test_spellings = train_test_split(tokenized_spelling_int, test_size=0.2, random_state=42)

# %%
# Tokenization:
# Transform the integer-coded words and phonemes into a format that can be used by the transformer model. You'll need to create tokenizers for both 
# the input (words) and output (phonemes) sequences. These tokenizers will convert your integer sequences into the required format, such as adding 
# special tokens like [CLS] and [SEP] for BERT or <s> and </s> for GPT.

import torch

# Assuming `padded_spelling_sequences` is your array of integer-encoded and padded spellings
encoder_inputs = train_spellings

# Remove the <end> token from the decoder input sequences
decoder_inputs = [seq[:-1] for seq in train_phonemes]

# Remove the <start> token from the decoder output sequences
decoder_outputs = [seq[1:] for seq in train_phonemes]


encoder_inputs = torch.tensor(encoder_inputs, dtype=torch.long)
decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
decoder_outputs = torch.tensor(decoder_outputs, dtype=torch.long)

# %% [markdown]
# ### Model training

# %%
# Create a dataset loader:
# Create a PyTorch or TensorFlow dataset loader to efficiently load the tokenized data during training. This will help you iterate through 
# the data in batches, allowing for better memory management and faster training times.

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class SpellingPhonemeDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_outputs):
        self.encoder_inputs = encoder_inputs
        self.decoder_inputs = decoder_inputs
        self.decoder_outputs = decoder_outputs

    def __len__(self):
        return len(self.encoder_inputs)

    def __getitem__(self, idx):
        return {
            'encoder_input': self.encoder_inputs[idx],
            'decoder_input': self.decoder_inputs[idx],
            'decoder_output': self.decoder_outputs[idx]
        }

spelling_phoneme_dataset = SpellingPhonemeDataset(encoder_inputs, decoder_inputs, decoder_outputs)

batch_size = 64  # You can set this to an appropriate value for your hardware
shuffle = True

spelling_phoneme_dataloader = DataLoader(
    spelling_phoneme_dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=0,  # You can set this to a higher value if your system supports multiple workers
    pin_memory=True,  # This can help improve data transfer performance if using GPU
    drop_last=True  # Drop the last incomplete batch
)


# %%
# Create embedding layers for the letters and phonemes:

import torch.nn as nn

num_letters = len(char_to_id)
num_phonemes = len(phoneme_to_id)

letter_embedding_dim = 50  # Set the desired dimension for the letter embeddings
phoneme_embedding_dim = 50  # Set the desired dimension for the phoneme embeddings

letter_embedding = nn.Embedding(num_letters, letter_embedding_dim)
phoneme_embedding = nn.Embedding(num_phonemes, phoneme_embedding_dim)

class CustomTransformer(nn.Module):
    def __init__(self, letter_embedding, phoneme_embedding, d_model, nhead, num_layers):
        super(CustomTransformer, self).__init__()
        self.letter_embedding = letter_embedding
        self.phoneme_embedding = phoneme_embedding
        
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc = nn.Linear(d_model, num_phonemes)
        
    def forward(self, encoder_inputs, decoder_inputs):
        # Apply the letter and phoneme embeddings
        embedded_encoder_inputs = self.letter_embedding(encoder_inputs)
        embedded_decoder_inputs = self.phoneme_embedding(decoder_inputs)

        # Process the embedded sequences with the transformer layers
        output = self.transformer(embedded_encoder_inputs.permute(1, 0, 2), embedded_decoder_inputs.permute(1, 0, 2))
        
        # Pass the transformer output through the final linear layer
        output = self.fc(output)

        return output.permute(1, 0, 2)


# %%
# Train your chosen transformer model on the prepared data. You'll need to define a suitable loss function, such as cross-entropy, 
# and an optimization algorithm, such as Adam, to update the model's parameters. During training, the model will learn to map input 
# sequences (spelling) to target sequences (phonemes).

import torch.optim as optim

d_model = 50
nhead = 5
num_layers = 2

custom_transformer = CustomTransformer(letter_embedding, phoneme_embedding, d_model, nhead, num_layers)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_transformer.to(device)

loss_function = nn.CrossEntropyLoss(ignore_index=0)  # 0 is the <pad> token's index
optimizer = optim.Adam(custom_transformer.parameters())

num_epochs = 100  # You can set this to the desired number of epochs for training

custom_transformer.train()

for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch in spelling_phoneme_dataloader:
        # Move the data to the GPU if available
        encoder_input_batch = batch['encoder_input'].to(device)
        decoder_input_batch = batch['decoder_input'].to(device)
        decoder_output_batch = batch['decoder_output'].to(device)

        # Forward pass
        output = custom_transformer(encoder_input_batch, decoder_input_batch)

        # Calculate the loss
        output = output.contiguous().view(-1, output.shape[-1])
        target = decoder_output_batch.view(-1)
        loss = loss_function(output, target)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update the model's parameters
        optimizer.step()

        epoch_loss += loss.item()

    # Print the average loss for this epoch
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(spelling_phoneme_dataloader)}")

# %% [markdown]
# ### Model evaluation

# %%
# Evaluate your model's performance on the validation set. Calculate metrics like accuracy, edit distance, or phoneme error rate
# to assess how well your model predicts the phonemes from the spelling of words. If the performance is not satisfactory, you can fine-tune 
# your model by adjusting its hyperparameters or architecture.

import csv

# Assuming `padded_spelling_sequences` is your array of integer-encoded and padded spellings
encoder_inputs = test_spellings

# Remove the <end> token from the decoder input sequences
decoder_inputs = [seq[:-1] for seq in test_phonemes]

# Remove the <start> token from the decoder output sequences
decoder_outputs = [seq[1:] for seq in test_phonemes]

encoder_inputs = torch.tensor(encoder_inputs, dtype=torch.long)
decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
decoder_outputs = torch.tensor(decoder_outputs, dtype=torch.long)

spelling_phoneme_dataset_test = SpellingPhonemeDataset(encoder_inputs, decoder_inputs, decoder_outputs)
spelling_phoneme_dataloader_test = DataLoader(spelling_phoneme_dataset_test, batch_size=batch_size, shuffle=False)

custom_transformer.eval()  # Set the model to evaluation mode
total_correct = 0
total_predictions = 0

with torch.no_grad():  # No need to compute gradients during evaluation
    for batch in spelling_phoneme_dataloader_test:
        # Move the data to the GPU if available
        encoder_input_batch = batch['encoder_input'].to(device)
        decoder_input_batch = batch['decoder_input'].to(device)
        decoder_output_batch = batch['decoder_output'].to(device)

        # Forward pass
        output = custom_transformer(encoder_input_batch, decoder_input_batch)
        predictions = output.argmax(dim=-1)

        # Calculate the number of correct predictions
        mask = (decoder_output_batch != 0)  # Ignore the <pad> tokens
        total_correct += (predictions[mask] == decoder_output_batch[mask]).sum().item()
        total_predictions += mask.sum().item()

accuracy = total_correct / total_predictions
print(f"Phoneme prediction accuracy: {accuracy:.4f}")

# Create reverse dictionaries for decoding
id_to_char = {idx: char for char, idx in char_to_id.items()}
id_to_phoneme = {idx: phoneme for phoneme, idx in phoneme_to_id.items()}

end_token_value = phoneme_to_id['</s>']
start_token_value = phoneme_to_id['<s>']
pad_token_value = phoneme_to_id['<pad>']
def int_to_text(sequence, int_to_token, end_token_value):
    return [int_to_token[int_val.item()] for int_val in sequence if int_val.item() != start_token_value and int_val.item() != end_token_value and int_val.item() != pad_token_value]

with open('transformer_phoneme_predictions.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Original Spelling', 'True Phonemes', 'Predicted Phonemes'])

    with torch.no_grad():  # No need to compute gradients during evaluation
        for i, batch in enumerate(spelling_phoneme_dataloader_test):
            # Move the data to the GPU if available
            encoder_input_batch = batch['encoder_input'].to(device)
            decoder_input_batch = batch['decoder_input'].to(device)
            decoder_output_batch = batch['decoder_output'].to(device)

            # Forward pass
            output = custom_transformer(encoder_input_batch, decoder_input_batch)
            predictions = output.argmax(dim=-1)

            # Convert integer-encoded sequences back to text
            for original, true_phonemes, predicted_phonemes in zip(encoder_input_batch, decoder_output_batch, predictions):
                original_text = ''.join(int_to_text(original, id_to_char, end_token_value))
                true_phonemes_text = ' '.join(int_to_text(true_phonemes, id_to_phoneme, end_token_value))
                predicted_phonemes_text = ' '.join(int_to_text(predicted_phonemes, id_to_phoneme, end_token_value))


                # Write the results to the CSV file
                csv_writer.writerow([original_text, true_phonemes_text, predicted_phonemes_text])


# %% [markdown]
# ### Clustering and cluster visualization

# %%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Initialize an empty matrix for storing cosine similarities
similarity_matrix = np.zeros((len(phoneme_to_id), len(char_to_id)))

# Calculate cosine similarities between all phoneme and letter embeddings
for i, phoneme in enumerate(id_to_phoneme.values()):
    for j, letter in enumerate(id_to_char.values()):
        phoneme_embedding = custom_transformer.phoneme_embedding.weight[i].detach().cpu().numpy()
        letter_embedding = custom_transformer.letter_embedding.weight[j].detach().cpu().numpy()
        similarity_matrix[i, j] = cosine_similarity([phoneme_embedding], [letter_embedding])[0, 0]

plt.figure(figsize=(12, 10))
sns.heatmap(similarity_matrix, cmap="coolwarm", xticklabels=id_to_char.values(), yticklabels=id_to_phoneme.values())
plt.xlabel("Letters")
plt.ylabel("Phonemes")
plt.title("Cosine Similarity between Phoneme and Letter Embeddings")
plt.show()

# %%
from sklearn.manifold import TSNE

# Combine the letter and phoneme embeddings into a single matrix
letter_embeddings = custom_transformer.letter_embedding.weight.detach().cpu().numpy()
phoneme_embeddings = custom_transformer.phoneme_embedding.weight.detach().cpu().numpy()
combined_embeddings = np.vstack((letter_embeddings, phoneme_embeddings))

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(combined_embeddings)

# Separate the 2D embeddings back into letters and phonemes
letter_embeddings_2d = embeddings_2d[:len(char_to_id), :]
phoneme_embeddings_2d = embeddings_2d[len(char_to_id):, :]

plt.figure(figsize=(12, 10))
plt.scatter(letter_embeddings_2d[:, 0], letter_embeddings_2d[:, 1], c='b', label='Letters')
plt.scatter(phoneme_embeddings_2d[:, 0], phoneme_embeddings_2d[:, 1], c='r', label='Phonemes')

# Annotate each point with its corresponding letter or phoneme
for i, letter in enumerate(id_to_char.values()):
    plt.annotate(letter, (letter_embeddings_2d[i, 0], letter_embeddings_2d[i, 1]))
    
for i, phoneme in enumerate(id_to_phoneme.values()):
    plt.annotate(phoneme, (phoneme_embeddings_2d[i, 0], phoneme_embeddings_2d[i, 1]))

plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.title("t-SNE Visualization of Letter and Phoneme Embeddings")
plt.legend()
plt.show()


# %%
from sklearn.cluster import KMeans

# Perform k-means clustering with 40 clusters
n_clusters = 40
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(phoneme_embeddings)

# Create an empty list to store the elements of each cluster
clusters = [[] for _ in range(n_clusters)]

# Iterate through the cluster labels and add the corresponding phonemes to the appropriate cluster
for i, label in enumerate(cluster_labels):
    clusters[label].append(list(id_to_phoneme.values())[i])

# Print the contents of each cluster
for i, cluster in enumerate(clusters):
    print(f"Cluster {i + 1}: {cluster}")

# %%
import string

human_readable_symbols = ['ↂ', '!', '⟕', '⫫', '⁑', '◬', '$', '#', '@', '<', 'ⵁ', '⭅', 'ⴼ', '⏃', '⸔', '⼺', '┳', 'Ⳃ', 'ⵖ', '❑', '⌒', '>', '‡', '☾', 'ⓧ', 'ⶻ', '⬗', 'Ɱ', '⽜', '╄', 'ⱬ', '⭘', '╪', 'ℤ', '▟', '⏊', '⼨', '⺁', '❶', 'ⓐ']

# Create a mapping from cluster index to human-readable symbols or single Latin letters
cluster_to_symbol = {}
latin_letters = set(string.ascii_uppercase + string.ascii_lowercase)
used_symbols = set()
for i, cluster_phoneme_list in enumerate(clusters):
    if len(cluster_phoneme_list) == 1 and cluster_phoneme_list[0] in latin_letters:
        cluster_to_symbol[i] = cluster_phoneme_list[0]
    else:
        for symbol in human_readable_symbols:
            if symbol not in used_symbols:
                cluster_to_symbol[i] = symbol
                used_symbols.add(symbol)
                break

symbol_to_phonemes = {}
for cluster, symbol in cluster_to_symbol.items():
    phonemes_in_cluster = clusters[cluster]
    symbol_to_phonemes[symbol] = phonemes_in_cluster

print(symbol_to_phonemes)

# %%
# Load the CMU Pronouncing Dictionary
cmu_dict = cmudict.dict()

# Define the input sentence
sentence = 'and maybe a snack for her brother Bob'  

# Split the sentence into words and convert them to uppercase
words = sentence.lower().split()

# Function to look up the phonemes for a word in the CMU Pronouncing Dictionary
def get_phonemes(word, cmu_dict):
    if word in cmu_dict:
        return cmu_dict[word][0]  # Use the first pronunciation variant
    else:
        return None

# Function to convert the phonemes to the new phonemes using the symbol mapping
def convert_phonemes(phonemes, symbol_to_phonemes):
    new_phonemes = []
    for phoneme in phonemes:
        for symbol, phoneme_list in symbol_to_phonemes.items():
            if phoneme in phoneme_list:
                new_phonemes.append(symbol)
                break
    return new_phonemes

# Iterate through the words, get their phonemes, and convert them to the new phonemes
new_sentence_phonemes = []
for word in words:
    phonemes = get_phonemes(word, cmu_dict)
    if phonemes:
        new_phonemes = convert_phonemes(phonemes, symbol_to_phonemes)
        new_sentence_phonemes.append(''.join(new_phonemes))
        print(f"Word: {word}")
        print(f"Original Phonemes: {' '.join(phonemes)}")
        print(f"New Phonemes: {''.join(new_phonemes)}")
        print()

# Print the converted phonemes for the input sentence with spaces between words only
print('Converted Sentence Phonemes:')
print(' '.join(new_sentence_phonemes))


# %%
print(new_sentence_phonemes)

# %%
latin = ['Please call Stella','Ask her to bring these things with her from the store','Six spoons of fresh snow peas','five thick slabs of blue cheese','and maybe a snack for her brother Bob']
natin = ['ⱬ╪⁑❑  KⴼL  !⸔A', '◬<K  #⁑  ⴼ⫫  ⴼ#⬗┳  ð⬗❑  $⬗┳❑  W⬗$  #⁑  F#AM  ð$  !#', '<⬗K<  <ⱬ⫫N❑  A⁑ⴼ  F#◬⏃  <Nⴼ  ⱬ@❑', 'F⟕ⴼ  $⬗K  <L◬ⴼ❑  A⁑ⴼ  ⴼL⫫  ⶻ@❑', '⸔⸔  Mℤⴼ⟕  A  <N◬K  F#  #⁑  ⴼ#A⁑ð⁑  ⴼ⁑ⴼ']

words = [word for seq in natin for word in seq.split()]
total_length = sum(len(word) for word in words)
average_length = total_length / len(words)
print(average_length)

# %%
latin_len = 4.088235294117647
natin_len = 3.1470588235294117
natin_len / latin_len


