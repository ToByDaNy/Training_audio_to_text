from sklearn.metrics import accuracy_score

from transformers import AutoProcessor, SeamlessM4Tv2Model, Trainer, EarlyStoppingCallback, TrainingArguments
import tensorflow as tf
import numpy as np
from transformers import pipeline, SeamlessM4Tv2Model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoProcessor, AutoModelForCTC
from IPython.display import Audio
from huggingface_hub import notebook_login
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from sklearn.model_selection import train_test_split

from huggingface_hub import HfApi, HfFolder

# Your token from the Hugging Face account
hf_token = ""

# Save the token to the Hugging Face cache folder
HfFolder.save_token(hf_token)

# Alternatively, you can set it directly in the environment variable (useful for scripts and CI/CD)
import os

os.environ["HF_TOKEN"] = hf_token

cv_17 = load_dataset("mozilla-foundation/common_voice_17_0", "ro", split='train[:20%]', trust_remote_code=True)


def preprocesare(batch):
    audio = batch["audio"]
    resampler = torchaudio.transforms.Resample(audio["sampling_rate"], 16_000)
    AR = resampler(torch.Tensor(audio["array"]))

    transcription = batch["sentence"]
    if transcription.startswith('"') and transcription.endswith('"'):
        transcription = transcription[1:-1]
    if transcription[-1] not in ['!', '.', '?']:
        transcription = transcription + '.'

    return {"sentence": transcription, "audio": AR}


data_set = cv_17.map(preprocesare, desc="Preprocesare ")

# Set seed for reproducibility
torch.manual_seed(1337)

# Load the processor and model
processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", src_lang="ron")
class AudioTextModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, audio, input_ids, attention_mask, labels=None):
        outputs = self.base_model(audio=audio, input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return outputs
model = AudioTextModel(SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large"))

# Extract sentences (text) and audio data
X_text = [sample["sentence"] for sample in data_set]
X_audio = [sample["audio"] for sample in data_set]

# Tokenize the text data
tokenizer = processor.tokenizer
encoded_texts = [tokenizer.encode(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128) for
                 text in X_text]

# Find the maximum length of audio samples
max_length = max(len(audio) for audio in X_audio)

# Pad the audio arrays to the same length
padded_audio_list = [np.pad(audio, (0, max_length - len(audio)), mode='constant', constant_values=0) for audio in
                     X_audio]
X_audio = np.stack(padded_audio_list)

# Convert encoded texts to the same length
encoded_texts = [torch.tensor(encoding, dtype=torch.long).squeeze() for encoding in encoded_texts]
max_text_length = max(len(encoding) for encoding in encoded_texts)
padded_texts = [
    np.pad(encoding, (0, max_text_length - len(encoding)), mode='constant', constant_values=tokenizer.pad_token_id) for
    encoding in encoded_texts]
X_text = np.stack(padded_texts)

# Create labels (y) for training
y = np.array(encoded_texts)

# Split the data into training and test sets
X_audio_train, X_audio_test, X_text_train, X_text_test, y_train, y_test = train_test_split(
    X_audio, X_text, y, test_size=0.2, random_state=42)

class AudioTextDataset(torch.utils.data.Dataset):
    def __init__(self, audio_data, text_data, labels):
        self.audio_data = torch.tensor(audio_data, dtype=torch.float32)
        self.text_data = torch.tensor(text_data, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.audio_data)

    def __getitem__(self, idx):
        return {
            "input_ids": self.text_data[idx],
            "audio": self.audio_data[idx],
            "labels": self.labels[idx]
        }
# Create Dataset instances
train_dataset = AudioTextDataset(X_audio_train, X_text_train, y_train)
test_dataset = AudioTextDataset(X_audio_test, X_text_test, y_test)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    logging_dir='./logs',
)

from transformers import DataCollatorWithPadding
from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class AudioTextDataCollator:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # First, pad the audio inputs
        audio_features = [{"input_features": feature["audio"]} for feature in features]
        audio_batch = self.processor.feature_extractor.pad(audio_features, return_tensors="pt")

        # Then, pad the text inputs
        text_features = [{"input_ids": feature["input_ids"]} for feature in features]
        text_batch = self.processor.tokenizer.pad(text_features, return_tensors="pt")

        # Combine audio and text features
        batch = {
            "audio": audio_batch["input_features"],
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
        }

        # Add labels if they're present
        if "labels" in features[0]:
            batch["labels"] = torch.stack([feature["labels"] for feature in features])

        return batch


data_collator = AudioTextDataCollator(processor)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=20)]
)
# Train the model
trainer.train()
model.save_pretrained("Model_Audio_to_Text_Seamless")
