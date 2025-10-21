# Transformer English-to-German Translation

## Overview
This project implements a Transformer-based sequence-to-sequence model for English-to-German translation using TensorFlow/Keras. It leverages the Multi30k dataset for training and evaluation. The model supports training, inference, and BLEU evaluation.

## Features

- Transformer model with multi-head attention, encoder-decoder layers.
- Automatic downloading and preprocessing of train and test datasets.
- Tokenization with SpaCy.
- BLEU score evaluation with NLTK.
- Sample translations after training for quick inspection.

## Requirements

- Python 3.10+
- TensorFlow 2.x
- NLTK
- SpaCy (en_core_web_sm and de_core_news_sm)


Create and activate the environemnt and install the dependencies using `requirements.txt`.

```bash
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

## Dataset

- Train set: Multi30k English-German sentences.
- Test set: Multi30k 2016 Flickr English-German sentences.

Automatically downloaded and preprocessed when running the script.

## Usage

Clone or download the repository.

Run the script:

```bash
python transformer_translation.py
# caffeinate -i python transformer_translation.py # Keep Mac awake during training
```

The script will:

- Download the datasets.
- Preprocess and tokenize the data.
- Train the Transformer model.
- Output training loss per epoch.
- Display sample translations.
- Evaluate BLEU score on the full test set.

## Notes

Training may take several hours depending on hardware.
- 300 epochs on MacBook Pro Apple M4 Pro Chip --> 09:36 – 15:50

The script uses a batch size of 32 by default.

Sample output:

```text
English: A man is playing a guitar.
German : Ein Mann spielt Gitarre.

English: Two dogs are running in the park.
German : Zwei Hunde laufen im Park.
```

BLEU score is printed at the end:
```text
Final BLEU score on full test set: 28.45
```

[Bilingual Evaluation Understudy (BLUE)](https://huggingface.co/spaces/evaluate-metric/bleu) is a metric that measures how closely a machine-generated translation matches one or more reference translations, based on overlapping n-grams. Scores range from 0 to 1 (or 0–100%), where higher values indicate better translations; roughly, scores above 30–40% are considered reasonable for full-sentence translations, while scores below 10–15% usually indicate poor quality.