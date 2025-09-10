# Install required packages
!pip install transformers torch datasets scikit-learn python-Levenshtein sentencepiece

import pandas as pd
import numpy as np
import re
import json
import torch
import os
import Levenshtein
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import (
    XLMRobertaTokenizer,
    XLMRobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import warnings
warnings.filterwarnings('ignore')

# Disable wandb
os.environ["WANDB_DISABLED"] = "true"

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class EnhancedThunglishPreprocessor:
    def __init__(self, levenshtein_threshold: float = 0.8, fuzzy_threshold: int = 2):
        """
        Enhanced preprocessor with Levenshtein distance-based fuzzy matching

        Args:
            levenshtein_threshold: Similarity threshold for fuzzy matching (0-1)
            fuzzy_threshold: Maximum edit distance for fuzzy matching
        """
        self.levenshtein_threshold = levenshtein_threshold
        self.fuzzy_threshold = fuzzy_threshold

        # Dictionary normalization (exact matches first)
        self.normalization_dict = {}

        # Cache for fuzzy matches to avoid repeated calculations
        self.fuzzy_cache = {}

        # Statistics tracking
        self.stats = {
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'no_matches': 0,
            'total_words': 0
        }

        # Enhanced normalization dictionary with more comprehensive mappings
        self.default_normalizations = {
            # Video and content related
            'vdo': 'video', 'vedio': 'video', 'vid': 'video', 'vids': 'videos',
            'vidz': 'videos', 'vidoe': 'video', 'vedios': 'videos',

            # Common abbreviations and internet slang
            'u': 'you', 'ur': 'your', 'r': 'are', 'n': 'and', 'nd': 'and',
            'thn': 'then', 'thx': 'thanks', 'ty': 'thank you', 'pls': 'please',
            'plz': 'please', 'gud': 'good', 'gd': 'good', 'gr8': 'great',
            'luv': 'love', 'lol': 'laugh out loud', 'omg': 'oh my god',
            'wtf': 'what the hell', 'btw': 'by the way', 'tbh': 'to be honest',
            'imo': 'in my opinion', 'imho': 'in my humble opinion',
            'fyi': 'for your information', 'asap': 'as soon as possible',

            # Numbers as words
            '2': 'to', '4': 'for', '8': 'ate',

            # Common contractions and informal speech
            'gonna': 'going to', 'wanna': 'want to', 'gotta': 'got to',
            'dunno': 'do not know', 'kinda': 'kind of', 'sorta': 'sort of',
            'outta': 'out of', 'lemme': 'let me', 'gimme': 'give me',
            'shoulda': 'should have', 'coulda': 'could have', 'woulda': 'would have',

            # Time related
            'mins': 'minutes', 'secs': 'seconds', 'hrs': 'hours',
            'yr': 'year', 'yrs': 'years', 'min': 'minute', 'sec': 'second', 'hr': 'hour',

            # Social media specific
            'subbed': 'subscribed', 'sub': 'subscribe', 'unsub': 'unsubscribe',
            'thumbs up': 'like', 'thumbs down': 'dislike', 'dm': 'direct message',
            'rt': 'retweet', 'fav': 'favorite', 'fb': 'facebook', 'ig': 'instagram',

            # Positive sentiment words (crucial for hope speech detection)
            'awsm': 'awesome', 'awsum': 'awesome', 'awsome': 'awesome',
            'awesum': 'awesome', 'amazng': 'amazing', 'amzing': 'amazing',
            'lovly': 'lovely', 'beautfl': 'beautiful', 'wonderfl': 'wonderful',
            'fantastc': 'fantastic', 'incredbl': 'incredible', 'excellnt': 'excellent',
            'perfct': 'perfect', 'marvlous': 'marvelous', 'magnfcnt': 'magnificent',

            # Hope and encouragement related
            'hopfl': 'hopeful', 'inspir': 'inspire', 'motivat': 'motivate',
            'positv': 'positive', 'optimstc': 'optimistic', 'encourag': 'encourage',
            'supportv': 'supportive', 'undrstnd': 'understand', 'togther': 'together',
            'strength': 'strength', 'courage': 'courage', 'resilient': 'resilient',
            'achieve': 'achieve', 'success': 'success', 'progress': 'progress',
            'improve': 'improve', 'growth': 'growth', 'knowledg': 'knowledge',
            'wisdom': 'wisdom', 'peace': 'peace', 'harmony': 'harmony',
            'respect': 'respect', 'dignity': 'dignity', 'freedom': 'freedom',
            'justice': 'justice', 'equality': 'equality', 'humanity': 'humanity',
            'future': 'future', 'dreams': 'dreams', 'goals': 'goals', 'hope': 'hope',

            # Common misspellings
            'recieve': 'receive', 'definately': 'definitely', 'seperate': 'separate',
            'occured': 'occurred', 'neccessary': 'necessary', 'accomodate': 'accommodate',
            'begining': 'beginning', 'existance': 'existence', 'priviledge': 'privilege',
            'wierd': 'weird', 'freind': 'friend', 'beleive': 'believe',
            'tommorrow': 'tomorrow', 'occassion': 'occasion', 'embarass': 'embarrass',

            # Stretched words (common in social media)
            'soooo': 'so', 'sooo': 'so', 'nooo': 'no', 'yesss': 'yes',
            'reallyyyy': 'really', 'niceee': 'nice', 'coool': 'cool',
            'hiii': 'hi', 'byeee': 'bye', 'thankss': 'thanks',
            'sorryyyy': 'sorry', 'pleaseee': 'please',

            # Tamil-English mixed words (common in Thunglish)
            'nanri': 'thanks', 'vanakkam': 'greetings', 'nalla': 'good',
            'super': 'super', 'mass': 'awesome', 'semma': 'awesome',
            'vera': 'very', 'level': 'awesome', 'adipoli': 'awesome',
            'powerfull': 'powerful', 'mindblowing': 'mind blowing',
        }

        # Add default normalizations
        self.normalization_dict.update(self.default_normalizations)

    def add_custom_normalizations(self, custom_dict: Dict[str, str]):
        """Add custom normalization mappings"""
        self.normalization_dict.update(custom_dict)

    def clean_word(self, word: str) -> str:
        """Extract the core alphabetic part of a word"""
        # Remove leading/trailing punctuation but keep internal structure
        cleaned = re.sub(r'^[^\w]+|[^\w]+$', '', word.lower())
        return cleaned

    def calculate_similarity(self, word1: str, word2: str) -> float:
        """Calculate Levenshtein similarity ratio between two words"""
        if len(word1) == 0 or len(word2) == 0:
            return 0.0

        distance = Levenshtein.distance(word1, word2)
        max_len = max(len(word1), len(word2))
        similarity = 1.0 - (distance / max_len)
        return similarity

    def find_fuzzy_match(self, word: str, candidates: List[str]) -> Optional[str]:
        """
        Find the best fuzzy match for a word from candidates using Levenshtein distance
        """
        if word in self.fuzzy_cache:
            return self.fuzzy_cache[word]

        best_match = None
        best_similarity = 0.0

        # Only consider candidates within reasonable length difference
        word_len = len(word)

        for candidate in candidates:
            candidate_len = len(candidate)

            # Skip if length difference is too large
            if abs(word_len - candidate_len) > self.fuzzy_threshold:
                continue

            # Calculate similarity
            similarity = self.calculate_similarity(word, candidate)

            # Check if this is a good match
            if (similarity >= self.levenshtein_threshold and
                similarity > best_similarity and
                Levenshtein.distance(word, candidate) <= self.fuzzy_threshold):
                best_match = candidate
                best_similarity = similarity

        # Cache the result
        self.fuzzy_cache[word] = best_match
        return best_match

    def normalize_word(self, word: str) -> str:
        """
        Normalize a single word using exact match first, then fuzzy matching
        """
        if not word or len(word) < 2:
            return word

        # Clean the word
        clean_word = self.clean_word(word)
        if not clean_word:
            return word

        self.stats['total_words'] += 1

        # Step 1: Exact match in dictionary
        if clean_word in self.normalization_dict:
            self.stats['exact_matches'] += 1
            normalized = self.normalization_dict[clean_word]

            # Preserve original case pattern
            if word.isupper():
                normalized = normalized.upper()
            elif word.istitle():
                normalized = normalized.title()
            elif word.islower():
                normalized = normalized.lower()

            # Replace the clean part while preserving punctuation
            return re.sub(re.escape(clean_word), normalized, word, flags=re.IGNORECASE)

        # Step 2: Fuzzy matching for OOV words
        candidates = list(self.normalization_dict.keys())
        fuzzy_match = self.find_fuzzy_match(clean_word, candidates)

        if fuzzy_match:
            self.stats['fuzzy_matches'] += 1
            normalized = self.normalization_dict[fuzzy_match]

            # Preserve case pattern
            if word.isupper():
                normalized = normalized.upper()
            elif word.istitle():
                normalized = normalized.title()
            elif word.islower():
                normalized = normalized.lower()

            # Replace the clean part while preserving punctuation
            return re.sub(re.escape(clean_word), normalized, word, flags=re.IGNORECASE)

        # Step 3: Handle stretched words (repeated characters)
        if len(clean_word) >= 3:
            # Remove excessive character repetition (keep max 2 of same char)
            destretched = re.sub(r'(.)\1{2,}', r'\1\1', clean_word)
            if destretched != clean_word and destretched in self.normalization_dict:
                self.stats['fuzzy_matches'] += 1
                normalized = self.normalization_dict[destretched]

                # Preserve case
                if word.isupper():
                    normalized = normalized.upper()
                elif word.istitle():
                    normalized = normalized.title()

                return re.sub(re.escape(clean_word), normalized, word, flags=re.IGNORECASE)

        # No match found
        self.stats['no_matches'] += 1
        return word

    def normalize_text(self, text: str) -> str:
        """
        Normalize text using hybrid approach: exact dictionary + fuzzy matching
        """
        if pd.isna(text):
            return ""

        text_str = str(text).strip()
        if not text_str:
            return ""

        # Split into words while preserving whitespace and structure
        words = text_str.split()
        normalized_words = []

        for word in words:
            normalized_word = self.normalize_word(word)
            normalized_words.append(normalized_word)

        return ' '.join(normalized_words)

    def preprocess_dataset(self, dataset: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Preprocess the entire dataset with enhanced normalization"""
        # Create a copy to avoid modifying original
        processed_df = dataset.copy()

        # Reset statistics
        self.stats = {
            'exact_matches': 0,
            'fuzzy_matches': 0,
            'no_matches': 0,
            'total_words': 0
        }

        # Add original text column for reference
        processed_df[f'{text_column}_original'] = processed_df[text_column]

        # Apply enhanced normalization
        print(f"Applying hybrid normalization to {len(processed_df)} texts...")
        processed_df[text_column] = processed_df[text_column].apply(self.normalize_text)

        # Remove empty texts
        initial_count = len(processed_df)
        processed_df = processed_df[processed_df[text_column].str.strip() != '']
        final_count = len(processed_df)

        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} empty texts")

        # Print normalization statistics
        self.print_normalization_stats()

        return processed_df

    def print_normalization_stats(self):
        """Print normalization statistics"""
        total = self.stats['total_words']
        if total > 0:
            print(f"\n📊 Normalization Statistics:")
            print(f"   Total words processed: {total}")
            print(f"   Exact matches: {self.stats['exact_matches']} ({self.stats['exact_matches']/total*100:.1f}%)")
            print(f"   Fuzzy matches: {self.stats['fuzzy_matches']} ({self.stats['fuzzy_matches']/total*100:.1f}%)")
            print(f"   No matches: {self.stats['no_matches']} ({self.stats['no_matches']/total*100:.1f}%)")
            print(f"   Total normalizations: {(self.stats['exact_matches'] + self.stats['fuzzy_matches'])/total*100:.1f}%")

    def build_vocabulary_from_data(self, texts: List[str], min_frequency: int = 2):
        """
        Build vocabulary from actual data to improve normalization coverage
        """
        print(f"Building vocabulary from {len(texts)} texts...")

        word_counter = Counter()
        for text in texts:
            if pd.notna(text):
                words = str(text).lower().split()
                for word in words:
                    clean_word = self.clean_word(word)
                    if len(clean_word) >= 2:
                        word_counter[clean_word] += 1

        # Find common words that might benefit from normalization
        common_words = [word for word, count in word_counter.items()
                       if count >= min_frequency and word not in self.normalization_dict]

        print(f"Found {len(common_words)} common words not in dictionary")

        # You could manually add some of these or use additional heuristics
        # For now, we'll just report them
        if len(common_words) > 0:
            print("Top 20 common words not in dictionary:")
            sorted_words = sorted([(word, word_counter[word]) for word in common_words[:20]],
                                key=lambda x: x[1], reverse=True)
            for word, count in sorted_words:
                print(f"  {word}: {count}")

class HopeSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # XLM-RoBERTa tokenizer handles multilingual text well
        # The text has already been normalized by our enhanced preprocessor
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            add_special_tokens=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions)
    }

# Dataset paths
train_path = f"/content/tamil_hope_first_train_restructured.csv"
val_path = f"/content/tamil_hope_first_dev_restructured.csv"
test_path = f"/content/tamil_hope_first_test_restructured.csv"

print("="*80)
print("ENHANCED THUNGLISH HOPE SPEECH DETECTION WITH XLM-ROBERTA")
print("="*80)

# Load datasets
print("\n1. Loading datasets...")
try:
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    print(f"Train dataset: {len(train_df)} samples")
    print(f"Validation dataset: {len(val_df)} samples")
    print(f"Test dataset: {len(test_df)} samples")

    # Display column names to understand structure
    print(f"\nDataset columns: {train_df.columns.tolist()}")
    print(f"Sample train data:")
    print(train_df.head(2))

except Exception as e:
    print(f"Error loading datasets: {e}")
    raise

# Identify text and label columns
possible_text_cols = ['text', 'comment', 'sentence', 'content', 'message']
possible_label_cols = ['label', 'class', 'category', 'hope_speech']

text_col = None
label_col = None

for col in possible_text_cols:
    if col in train_df.columns:
        text_col = col
        break

for col in possible_label_cols:
    if col in train_df.columns:
        label_col = col
        break

if text_col is None or label_col is None:
    print(f"Could not identify text and label columns. Available columns: {train_df.columns.tolist()}")
    # Use the first text-like column and first numeric column as fallback
    text_col = train_df.select_dtypes(include=['object']).columns[0]
    label_col = train_df.select_dtypes(include=['int64', 'float64']).columns[0]
    print(f"Using text_col: '{text_col}', label_col: '{label_col}'")

print(f"\nUsing text column: '{text_col}'")
print(f"Using label column: '{label_col}'")

# Check label distribution
print(f"\nLabel distribution in training data:")
print(train_df[label_col].value_counts())

# Create label mapping
unique_labels = train_df[label_col].unique().tolist()
label_mapping = {label: i for i, label in enumerate(unique_labels)}
inverse_label_mapping = {i: label for label, i in label_mapping.items()}

print(f"\nLabel mapping: {label_mapping}")

# Initialize enhanced preprocessor with fuzzy matching
print("\n2. Initializing enhanced text preprocessor with Levenshtein fuzzy matching...")
preprocessor = EnhancedThunglishPreprocessor(
    levenshtein_threshold=0.8,  # 80% similarity threshold
    fuzzy_threshold=2  # Maximum edit distance of 2
)

# Add domain-specific custom normalizations
custom_normalizations = {
    # YouTube/Social media specific
    'youtuber': 'youtuber', 'ytber': 'youtuber', 'yter': 'youtuber',
    'subscriber': 'subscriber', 'subcriber': 'subscriber',
    'channel': 'channel', 'chanl': 'channel', 'chanel': 'channel',

    # Hope speech specific terms
    'inspiring': 'inspiring', 'inspirng': 'inspiring', 'inspirin': 'inspiring',
    'motivating': 'motivating', 'motivatng': 'motivating', 'motivatin': 'motivating',
    'encouraging': 'encouraging', 'encouragng': 'encouraging', 'encouragin': 'encouraging',
    'supportive': 'supportive', 'suportive': 'supportive', 'supportiv': 'supportive',

    # Tamil-English mixed expressions
    'vanakkam': 'greetings', 'vanakam': 'greetings', 'vanakkam': 'greetings',
    'nanri': 'thanks', 'nandri': 'thanks', 'thank you': 'thank you',
    'semma': 'awesome', 'samma': 'awesome', 'vera level': 'very good',
    'mass': 'awesome', 'masss': 'awesome', 'massss': 'awesome',
    'adipoli': 'awesome', 'adipoli': 'excellent', 'superb': 'superb',

    # Common positive expressions in Thunglish
    'nalla': 'good', 'nala': 'good', 'nallla': 'good',
    'super': 'super', 'supeer': 'super', 'supper': 'super',
    'excellent': 'excellent', 'excelent': 'excellent', 'excellnt': 'excellent',
    'wonderful': 'wonderful', 'wonderfull': 'wonderful', 'wondeful': 'wonderful',
    'beautiful': 'beautiful', 'beautifull': 'beautiful', 'beatiful': 'beautiful',
    'amazing': 'amazing', 'amazingg': 'amazing', 'amazng': 'amazing',
    'fantastic': 'fantastic', 'fantatic': 'fantastic', 'fantasic': 'fantastic',
}

preprocessor.add_custom_normalizations(custom_normalizations)

# Build vocabulary from training data to improve coverage
print("\n3. Analyzing vocabulary from training data...")
all_train_texts = train_df[text_col].dropna().tolist()
preprocessor.build_vocabulary_from_data(all_train_texts, min_frequency=3)

# Preprocess datasets with enhanced normalization
print("\n4. Preprocessing datasets with hybrid normalization...")
train_df_processed = preprocessor.preprocess_dataset(train_df, text_col)
val_df_processed = preprocessor.preprocess_dataset(val_df, text_col)
test_df_processed = preprocessor.preprocess_dataset(test_df, text_col)

print("\n🔍 Sample of enhanced preprocessing results:")
for i in range(min(5, len(train_df_processed))):
    original = train_df_processed.iloc[i][f'{text_col}_original']
    processed = train_df_processed.iloc[i][text_col]
    label = train_df_processed.iloc[i][label_col]

    print(f"\n--- Sample {i+1} ---")
    print(f"Original:  {original}")
    print(f"Processed: {processed}")
    print(f"Label:     {label}")
    print(f"Changed:   {'✓' if original != processed else '✗'}")

# Initialize XLM-RoBERTa tokenizer and model
print("\n5. Loading XLM-RoBERTa model and tokenizer...")
model_name = "xlm-roberta-base"
tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)

# Get number of unique labels
num_labels = len(label_mapping)
print(f"Number of classes: {num_labels}")

model = XLMRobertaForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
)

# Move model to device
model.to(device)

print(f"✅ Loaded XLM-RoBERTa model with {num_labels} classes")
print(f"🌍 XLM-RoBERTa supports 100+ languages including Tamil and English")
print(f"📊 Model parameters: ~270M (base model)")

# Convert labels to integers
train_labels_encoded = train_df_processed[label_col].map(label_mapping).tolist()
val_labels_encoded = val_df_processed[label_col].map(label_mapping).tolist()
test_labels_encoded = test_df_processed[label_col].map(label_mapping).tolist()

# Create datasets
print("\n6. Creating PyTorch datasets...")
train_dataset = HopeSpeechDataset(
    train_df_processed[text_col].tolist(),
    train_labels_encoded,
    tokenizer,
    max_length=512
)

val_dataset = HopeSpeechDataset(
    val_df_processed[text_col].tolist(),
    val_labels_encoded,
    tokenizer,
    max_length=512
)

test_dataset = HopeSpeechDataset(
    test_df_processed[text_col].tolist(),
    test_labels_encoded,
    tokenizer,
    max_length=512
)

# Training arguments optimized for XLM-RoBERTa
print("\n7. Setting up training with XLM-RoBERTa...")
training_args = TrainingArguments(
    output_dir='./xlm_roberta_results',
    num_train_epochs=15,  # Slightly fewer epochs as XLM-RoBERTa is more powerful
    per_device_train_batch_size=8,  # Smaller batch size due to larger model
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,  # To maintain effective batch size
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./xlm_roberta_logs',
    logging_steps=100,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=3,
    seed=42,
    fp16=torch.cuda.is_available(),
    report_to=[],
    dataloader_num_workers=2,
    lr_scheduler_type="linear",
    learning_rate=1e-5,  # Lower learning rate for XLM-RoBERTa
)

# Initialize trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Train model
print("\n8. Training XLM-RoBERTa model with enhanced preprocessing...")
print("🚀 Training will benefit from:")
print("   • XLM-RoBERTa's multilingual capabilities (Tamil + English)")
print("   • Exact dictionary normalization for known variations")
print("   • Levenshtein fuzzy matching for spelling mistakes and noise")
print("   • RoBERTa's improved training methodology over BERT")
print("   • Better handling of code-mixed (Thunglish) text")
print(f"   • Early stopping with patience=3 epochs")

try:
    trainer.train()
    print("\n✅ XLM-RoBERTa training completed successfully!")
except Exception as e:
    print(f"❌ Training error: {e}")

# Evaluate on test set
print("\n9. Evaluating XLM-RoBERTa on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test Results: {test_results}")

# Get detailed predictions
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = test_labels_encoded

print("\n📈 Detailed XLM-RoBERTa Test Results:")
print(f"Test Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(inverse_label_mapping.values())))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save XLM-RoBERTa model
print("\n10. Saving XLM-RoBERTa model and artifacts...")
model.save_pretrained('./xlm_roberta_hope_speech_model')
tokenizer.save_pretrained('./xlm_roberta_hope_speech_model')

# Save processed datasets
train_df_processed.to_csv('train_xlm_roberta_processed.csv', index=False)
val_df_processed.to_csv('val_xlm_roberta_processed.csv', index=False)
test_df_processed.to_csv('test_xlm_roberta_processed.csv', index=False)

# Save enhanced preprocessing artifacts
preprocessing_artifacts = {
    'model_type': 'xlm-roberta-base',
    'normalization_dict': preprocessor.normalization_dict,
    'label_mapping': label_mapping,
    'levenshtein_threshold': preprocessor.levenshtein_threshold,
    'fuzzy_threshold': preprocessor.fuzzy_threshold,
    'normalization_stats': preprocessor.stats,
    'fuzzy_cache_size': len(preprocessor.fuzzy_cache)
}

with open('xlm_roberta_preprocessing_artifacts.json', 'w', encoding='utf-8') as f:
    json.dump(preprocessing_artifacts, f, indent=2, ensure_ascii=False)

print("✅ XLM-RoBERTa model and artifacts saved!")

# Final results summary
final_results = {
    'test_accuracy': accuracy_score(y_true, y_pred),
    'model_type': 'xlm-roberta-base',
    'normalization_stats': preprocessor.stats,
    'num_samples': {
        'train': len(train_df_processed),
        'val': len(val_df_processed),
        'test': len(test_df_processed)
    }
}

print(f"\n🎯 FINAL RESULTS WITH XLM-ROBERTA + ENHANCED PREPROCESSING:")
print(f"🎯 Test Accuracy: {final_results['test_accuracy']:.4f}")
print(f"🤖 Model: {final_results['model_type']}")
print(f"📊 Training samples: {final_results['num_samples']['train']}")
print(f"📊 Validation samples: {final_results['num_samples']['val']}")
print(f"📊 Test samples: {final_results['num_samples']['test']}")

print(f"\n🔧 PREPROCESSING IMPACT:")
total_words = final_results['normalization_stats']['total_words']
total_normalizations = (final_results['normalization_stats']['exact_matches'] +
                       final_results['normalization_stats']['fuzzy_matches'])
normalization_rate = (total_normalizations / total_words * 100) if total_words > 0 else 0

print(f"📝 Total words processed: {total_words:,}")
print(f"🎯 Words normalized: {total_normalizations:,} ({normalization_rate:.1f}%)")
print(f"📚 Exact dictionary matches: {final_results['normalization_stats']['exact_matches']:,}")
print(f"🔍 Fuzzy matches (Levenshtein): {final_results['normalization_stats']['fuzzy_matches']:,}")
print(f"❌ No matches found: {final_results['normalization_stats']['no_matches']:,}")

print(f"\n💾 Saved artifacts:")
print(f"   • XLM-RoBERTa model: './xlm_roberta_hope_speech_model/'")
print(f"   • Processed datasets: train/val/test_xlm_roberta_processed.csv")
print(f"   • Preprocessing config: xlm_roberta_preprocessing_artifacts.json")
print(f"   • Fuzzy cache entries: {len(preprocessor.fuzzy_cache):,}")

print("\n🎉 XLM-RoBERTa Hope Speech Detection Pipeline Completed Successfully!")
print("🚀 The model now benefits from:")
print("   ✓ XLM-RoBERTa's multilingual training (100+ languages)")
print("   ✓ Native Tamil language understanding")
print("   ✓ Comprehensive dictionary normalization")
print("   ✓ Levenshtein distance-based fuzzy matching")
print("   ✓ Intelligent handling of code-mixed (Thunglish) text")
print("   ✓ RoBERTa's improved training methodology")
print("   ✓ Better handling of social media text variations")

# Additional analysis: Show some examples of fuzzy matches found
print(f"\n🔍 FUZZY MATCHING EXAMPLES:")
if preprocessor.fuzzy_cache:
    fuzzy_examples = [(k, v) for k, v in preprocessor.fuzzy_cache.items() if v is not None]
    if fuzzy_examples:
        print("   Words corrected through fuzzy matching:")
        for i, (original, corrected) in enumerate(fuzzy_examples[:10]):  # Show first 10
            similarity = preprocessor.calculate_similarity(original, corrected)
            print(f"   • '{original}' → '{preprocessor.normalization_dict[corrected]}' (similarity: {similarity:.2f})")

        if len(fuzzy_examples) > 10:
            print(f"   ... and {len(fuzzy_examples) - 10} more fuzzy corrections")
    else:
        print("   No fuzzy matches were applied in this dataset")
else:
    print("   Fuzzy matching cache is empty")

print(f"\n📊 XLM-ROBERTA MODEL PERFORMANCE SUMMARY:")
print(f"   Final Test Accuracy: {final_results['test_accuracy']:.4f}")
print(f"   Training Samples: {final_results['num_samples']['train']:,}")
print(f"   Model Type: XLM-RoBERTa Base (~270M parameters)")
print(f"   Max Sequence Length: 512 tokens")
print(f"   Preprocessing: Hybrid (Dictionary + Levenshtein Fuzzy)")
print(f"   Multilingual Support: 100+ languages including Tamil")
print(f"   Architecture: RoBERTa (Robustly Optimized BERT)")

# XLM-RoBERTa specific advantages
print(f"\n🌍 XLM-ROBERTA ADVANTAGES FOR THUNGLISH:")
print(f"   • Pre-trained on Tamil and English data")
print(f"   • Better handling of code-switching between languages")
print(f"   • Improved subword tokenization for mixed scripts")
print(f"   • No Next Sentence Prediction (NSP) - focus on MLM")
print(f"   • Dynamic masking during training")
print(f"   • Better performance on downstream tasks")

# Performance comparison insight
print(f"\n📈 EXPECTED IMPROVEMENTS OVER DISTILBERT:")
print(f"   • Better multilingual understanding")
print(f"   • Improved handling of Tamil-English code-mixing")
print(f"   • More robust to spelling variations")
print(f"   • Better contextual embeddings for hope speech detection")

# Save comprehensive final summary
summary_report = {
    "model_performance": {
        "test_accuracy": final_results['test_accuracy'],
        "model_type": "XLM-RoBERTa Base",
        "model_parameters": "~270M",
        "max_sequence_length": 512,
        "languages_supported": "100+ including Tamil and English"
    },
    "preprocessing_stats": final_results['normalization_stats'],
    "dataset_info": final_results['num_samples'],
    "fuzzy_matching": {
        "threshold": preprocessor.levenshtein_threshold,
        "max_edit_distance": preprocessor.fuzzy_threshold,
        "cache_size": len(preprocessor.fuzzy_cache)
    },
    "xlm_roberta_advantages": [
        "Multilingual pre-training",
        "Native Tamil support",
        "Code-switching handling",
        "RoBERTa improvements over BERT",
        "Dynamic masking",
        "No NSP task"
    ]
}

with open('xlm_roberta_final_report.json', 'w', encoding='utf-8') as f:
    json.dump(summary_report, f, indent=2, ensure_ascii=False)

print(f"\n📋 Complete XLM-RoBERTa model report saved to: 'xlm_roberta_final_report.json'")

# Function to test the model on new examples
def test_model_prediction(text_sample):
    """Test the trained model on a new text sample"""
    # Preprocess the text
    processed_text = preprocessor.normalize_text(text_sample)

    # Tokenize
    inputs = tokenizer(
        processed_text,
        truncation=True,
        padding='max_length',
        max_length=512,
        return_tensors='pt'
    )

    # Move to device
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1).item()
        confidence = torch.max(predictions).item()

    return {
        'original_text': text_sample,
        'processed_text': processed_text,
        'predicted_class': inverse_label_mapping[predicted_class],
        'confidence': confidence,
        'all_probabilities': {inverse_label_mapping[i]: prob.item()
                            for i, prob in enumerate(predictions[0])}
    }

print(f"\n🧪 TESTING MODEL ON SAMPLE TEXTS:")
# Test examples (you can modify these)
test_examples = [
    "This video is vera level mass! Super content bro, keep inspiring us 🔥",
    "Nalla content da, very motivating and hopeful for young generation",
    "Semma video! This gives me hope and courage to achieve my dreams",
    "Beautiful message, nanri for sharing such positive thoughts",
    "Amazing work! This channel always encourages and supports everyone"
]

for i, example in enumerate(test_examples):
    result = test_model_prediction(example)
    print(f"\n--- Test Example {i+1} ---")
    print(f"Original: {result['original_text']}")
    print(f"Processed: {result['processed_text']}")
    print(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.3f})")
    print(f"All probabilities: {result['all_probabilities']}")

print("="*80)
print("🎊 XLM-ROBERTA ENHANCED THUNGLISH HOPE SPEECH DETECTION COMPLETED!")
print("="*80)
