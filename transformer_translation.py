# ------------------------------
# Required libraries 
# ------------------------------

import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from transformer_layers.layers import EncoderLayer, DecoderLayer
from nltk.translate.bleu_score import corpus_bleu

from utils.file_utils import download_file, save_tokenized_data, save_encoded_data
from utils.model_utils import get_positional_encoding
from utils.nlp_utils import setup_nlp

spacy_en, spacy_de = setup_nlp()

# ------------------------------
# Download data 
# ------------------------------

# Download train set
download_file("https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.en.gz", "train.en")
download_file("https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.de.gz", "train.de")

with open("data/raw/train.en", "r", encoding="utf-8") as f:
    train_en_sentences = [line.strip() for line in f]
with open("data/raw/train.de", "r", encoding="utf-8") as f:
    train_de_sentences = [line.strip() for line in f]

train_data = list(zip(train_en_sentences, train_de_sentences))

# Download test set
download_file("https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.en.gz", "test.en")
download_file("https://github.com/multi30k/dataset/raw/master/data/task1/raw/test_2016_flickr.de.gz", "test.de")

with open("data/raw/test.en", "r", encoding="utf-8") as f:
    test_en_sentences = [line.strip() for line in f]
with open("data/raw/test.de", "r", encoding="utf-8") as f:
    test_de_sentences = [line.strip() for line in f]

test_data = list(zip(test_en_sentences, test_de_sentences))


# ------------------------------
# Preprocess data
# ------------------------------
def tokenize_en(text): return [tok.text.lower() for tok in spacy_en.tokenizer(text)]
def tokenize_de(text): return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

train_en_tokens = [tokenize_en(s) for s in train_en_sentences]
train_de_tokens = [tokenize_de(s) for s in train_de_sentences]

save_tokenized_data(train_en_tokens, "data/tokenized/train.en.tok")
save_tokenized_data(train_de_tokens, "data/tokenized/train.de.tok")

def build_vocab(tokenizer, sentences, min_freq=2):
    vocab = {"<pad>":0, "<bos>":1, "<eos>":2, "<unk>":3}
    idx = 4
    freq = {}
    for sentence in sentences:
        for token in tokenizer(sentence):
            freq[token] = freq.get(token,0)+1
    for token, count in freq.items():
        if count >= min_freq:
            vocab[token] = idx
            idx += 1
    return vocab

SRC_VOCAB = build_vocab(tokenize_en, train_en_sentences, min_freq=1)
TGT_VOCAB = build_vocab(tokenize_de, train_de_sentences, min_freq=1)

SRC_PAD_IDX = SRC_VOCAB["<pad>"]
TGT_PAD_IDX = TGT_VOCAB["<pad>"]

# ------------------------------
# Dataset creation
# ------------------------------
def encode_sentence(vocab, tokenizer, sentence):
    tokens = ["<bos>"] + tokenizer(sentence) + ["<eos>"]
    return [vocab.get(tok, vocab["<unk>"]) for tok in tokens]

def create_dataset(data, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, batch_size=32):
    src_seqs, tgt_seqs = [], []
    for src, tgt in data:
        src_seqs.append(encode_sentence(src_vocab, src_tokenizer, src))
        tgt_seqs.append(encode_sentence(tgt_vocab, tgt_tokenizer, tgt))
    src_seqs = pad_sequences(src_seqs, padding='post', value=SRC_PAD_IDX)
    tgt_seqs = pad_sequences(tgt_seqs, padding='post', value=TGT_PAD_IDX)
    return tf.data.Dataset.from_tensor_slices((src_seqs, tgt_seqs)).shuffle(1024).batch(batch_size)

train_dataset = create_dataset(train_data, SRC_VOCAB, TGT_VOCAB, tokenize_en, tokenize_de)

encoded_sentences   = [encode_sentence(SRC_VOCAB, tokenize_en, s) for s in train_en_sentences]
save_encoded_data(encoded_sentences, "data/encoded/train_en_encoded.txt")

encoded_sentences   = [encode_sentence(TGT_VOCAB, tokenize_de, s) for s in train_de_sentences]
save_encoded_data(encoded_sentences, "data/encoded/train_de_encoded.txt")

# ------------------------------
# Transformer
# ------------------------------
class Transformer(Model):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, dff=2048, max_seq_len=50, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = Embedding(src_vocab_size, d_model)
        self.tgt_embedding = Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = get_positional_encoding(max_seq_len, d_model)
        self.enc_layers = [EncoderLayer(d_model,num_heads,dff,dropout) for _ in range(num_layers)]
        self.dec_layers = [DecoderLayer(d_model,num_heads,dff,dropout) for _ in range(num_layers)]
        self.final_layer = Dense(tgt_vocab_size)

    def call(self, src, tgt, training):
        seq_len = tf.shape(src)[1]
        src_emb = self.src_embedding(src) + self.pos_encoding[:,:seq_len,:]
        enc_output = src_emb
        for layer in self.enc_layers:
            enc_output = layer(enc_output, training=training)

        tgt_seq_len = tf.shape(tgt)[1]
        tgt_emb = self.tgt_embedding(tgt) + self.pos_encoding[:,:tgt_seq_len,:]
        dec_output = tgt_emb
        for layer in self.dec_layers:
            dec_output = layer(dec_output, enc_output, training=training)
        return self.final_layer(dec_output)

# ------------------------------
# Training
# ------------------------------
transformer = Transformer(len(SRC_VOCAB), len(TGT_VOCAB))
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
optimizer = Adam(0.001)

def loss_function(real, pred):
    mask = tf.cast(tf.not_equal(real, TGT_PAD_IDX), tf.float32)
    loss_ = loss_object(real, pred)
    return tf.reduce_sum(loss_ * mask)/tf.reduce_sum(mask)

@tf.function
def train_step(src, tgt):
    tgt_inp = tgt[:,:-1]
    tgt_real = tgt[:,1:]
    with tf.GradientTape() as tape:
        predictions = transformer(src, tgt_inp, training=True)
        loss = loss_function(tgt_real, predictions)
    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    return loss

EPOCHS = 8
for epoch in range(EPOCHS):
    total_loss = 0
    for batch, (src, tgt) in enumerate(train_dataset):
        batch_loss = train_step(src, tgt)
        total_loss += batch_loss
    print(f"Epoch {epoch+1}, Loss: {total_loss / (batch+1):.4f}")

# ------------------------------
# Evaluation
# ------------------------------
def translate_sentence_tf(model, sentence, src_vocab, tgt_vocab, max_len=50):
    tokens = ["<bos>"] + tokenize_en(sentence) + ["<eos>"]
    src_indices = [src_vocab.get(tok, src_vocab["<unk>"]) for tok in tokens]
    src_tensor = tf.expand_dims(src_indices, 0)
    tgt_indices = [tgt_vocab["<bos>"]]
    for i in range(max_len):
        tgt_tensor = tf.expand_dims(tgt_indices, 0)
        preds = model(src_tensor, tgt_tensor, training=False)
        next_id = tf.argmax(preds[0,-1,:]).numpy()
        tgt_indices.append(next_id)
        if next_id == tgt_vocab["<eos>"]:
            break
    inv_tgt_vocab = {v:k for k,v in tgt_vocab.items()}
    return " ".join([inv_tgt_vocab[i] for i in tgt_indices[1:-1]])

sample_sentences = [
    "The AI model predicts tomorrow’s energy demand.",
    "Solar panels are generating power efficiently.",
    "The system schedules maintenance for wind turbines.",
    "The project manager reviews the power grid report.",
    "Business analysts plan investments in renewable energy."
]

print("\n--- Sample Translations ---\n")
for sentence in sample_sentences:
    translation = translate_sentence_tf(transformer, sentence, SRC_VOCAB, TGT_VOCAB)
    print(f"English: {sentence}")
    print(f"German : {translation}\n")

# ------------------------------
# BLEU evaluation 
# ------------------------------

subset_size = 50
demo_test_data = test_data[:subset_size]  # take first 50 examples

references = []
hypotheses = []

for src_sent, tgt_sent in demo_test_data:
    translation = translate_sentence_tf(transformer, src_sent, SRC_VOCAB, TGT_VOCAB)
    hypotheses.append(translation.split())
    references.append([tokenize_de(tgt_sent)])

bleu_score = corpus_bleu(references, hypotheses)
print(f"\nBLEU score on small subset ({subset_size} sentences): {bleu_score*100:.2f}")

