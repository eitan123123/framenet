import spacy
import nltk
from nltk.corpus import framenet as fn
from typing import Tuple, List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np

# Global setup
nlp = spacy.load("en_core_web_sm")
MAX_LENGTH = 512


class FrameAnalyzer:
    def __init__(self, model_name="microsoft/deberta-base-mnli", method="advanced"):
        """
        Initialize with either microsoft/deberta-base-mnli or
        MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.method = method
        self.is_microsoft = "microsoft" in model_name.lower()

    def get_first_sentence(self, text: str) -> str:
        """Extract the complete first sentence without truncation"""
        # First, clean up any parenthetical examples
        text = text.split("(e.g.")[0].split("(e")[0].strip()

        # Get first complete sentence
        sentences = text.split('.')
        first_sent = sentences[0].strip().lower()

        # Remove any dangling parentheses
        first_sent = first_sent.split('(')[0].strip()

        # Add period if not present
        if not first_sent.endswith('.'):
            first_sent += '.'

        return first_sent

    def create_natural_hypothesis_basic(self, sentence: str, frame: Dict[str, str]) -> str:
        """Basic method: simple concatenation with frame definition"""
        first_sentence = self.get_first_sentence(frame["definition"])
        return f"This is an event where {first_sentence}"

    def create_natural_hypothesis_advanced(self, sentence: str, frame: Dict[str, str]) -> str:
        """Advanced method: more sophisticated hypothesis creation"""
        doc = nlp(sentence)
        subject = next((token for token in doc if token.dep_ == "nsubj"), None)
        verb = next((token for token in doc if token.pos_ == "VERB"), None)
        direct_object = next((token for token in doc if token.dep_ == "dobj"), None)

        first_sentence = self.get_first_sentence(frame["definition"])

        hypotheses = []
        if subject and direct_object:
            hypotheses.append(
                f"{subject.text} and {direct_object.text} are involved in an event where {first_sentence}"
            )

        prep_phrases = []
        for token in doc:
            if token.dep_ == "prep":
                pobj = next((child for child in token.children if child.dep_ == "pobj"), None)
                if pobj:
                    prep_phrases.append(f"{token.text} {pobj.text}")

        if prep_phrases:
            prep_context = " ".join(prep_phrases)
            hypotheses.append(f"An action occurs {prep_context} that involves {first_sentence}")

        hypotheses.append(f"The described situation involves {first_sentence}")

        if verb and direct_object:
            hypotheses.append(f"The {direct_object.text} undergoes an action where {first_sentence}")

        hypotheses = list(dict.fromkeys([h for h in hypotheses if h]))
        return max(hypotheses, key=lambda h: self.calculate_semantic_similarity(sentence, h))

    def calculate_semantic_similarity(self, premise: str, hypothesis: str) -> float:
        """Calculate semantic similarity based on model type"""
        inputs = self.tokenizer(premise, hypothesis,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=MAX_LENGTH)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        if self.is_microsoft:
            # Microsoft/deberta-base-mnli: use contradiction score (index 2)
            return scores[0][2].item()
        else:
            # MoritzLaurer model: use entailment score (index 0)
            return scores[0][0].item()

    def calculate_nli_entailment_score(self, premise: str, hypothesis: str) -> float:
        """Additional method for explicit entailment scoring with MoritzLaurer model"""
        if self.is_microsoft:
            raise ValueError("Entailment scoring not applicable for Microsoft model")

        inputs = self.tokenizer(premise, hypothesis,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=MAX_LENGTH)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        return scores[0][0].item()

    def find_relevant_frames(self, verb: str) -> List[Dict[str, str]]:
        """Find frames relevant to the given verb"""
        frames = []
        verb_search = verb.lower()

        for frame in fn.frames():
            for lu in frame.lexUnit.keys():
                if verb_search in lu.lower() and '.v' in lu.lower():
                    frame_info = {
                        "name": frame.name,
                        "definition": frame.definition
                    }
                    frames.append(frame_info)
                    break
        return frames

    def process_dataset(self, input_csv: str, output_csv: str):
        """Process dataset with evaluation metrics"""
        df = pd.read_csv(input_csv)

        # Initialize columns
        new_columns = ['Frame Definition', 'Notes', 'Did the model Predicte the Frame']
        for col in new_columns:
            if col not in df.columns:
                df[col] = ''

        # Convert columns to string type
        string_columns = ['Envoked Frame', 'Ground Truth (Manually Checking)',
                          'Frame Definition', 'Notes', 'Did the model Predicte the Frame']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype('string')

        results_dict = {}
        predictions = []
        ground_truths = []

        # Process unique sentences
        unique_sentences = df.drop_duplicates(subset=['ID', 'Sentence'])

        for _, row in unique_sentences.iterrows():
            sentence_id = row['ID']
            sentence = row['Sentence']
            verb = row['Verb']

            if sentence_id not in results_dict:
                frames = self.find_relevant_frames(verb)
                results = []

                for frame in frames:
                    hypothesis = (self.create_natural_hypothesis_advanced(sentence, frame)
                                  if self.method == "advanced"
                                  else self.create_natural_hypothesis_basic(sentence, frame))

                    similarity = self.calculate_semantic_similarity(sentence, hypothesis)
                    results.append({
                        "frame": frame["name"],
                        "definition": frame["definition"],
                        "similarity_score": similarity
                    })

                results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
                results_dict[sentence_id] = results

        # Update DataFrame and collect metrics
        for idx, row in df.iterrows():
            sentence_id = row['ID']
            frame_num = row['Frame #']
            ground_truth = row['Ground Truth (Manually Checking)']

            if sentence_id in results_dict:
                results = results_dict[sentence_id][:12]  # Take top 12 frames

                if frame_num <= len(results):
                    frame_result = results[frame_num - 1]
                    confidence = frame_result['similarity_score']

                    df.at[idx, 'Envoked Frame'] = frame_result['frame']
                    df.at[idx, 'Frame Definition'] = frame_result['definition']
                    df.at[idx, 'Notes'] = f"Confidence: {confidence:.3f}"

                    prediction = 'yes' if confidence >= 0.7 else 'no'
                    df.at[idx, 'Did the model Predicte the Frame'] = 'v' if prediction == 'yes' else ''

                    if ground_truth and ground_truth.strip().lower() in ['yes', 'no']:
                        predictions.append(prediction)
                        ground_truths.append(ground_truth.strip().lower())
                else:
                    df.at[idx, 'Envoked Frame'] = ''
                    df.at[idx, 'Frame Definition'] = ''
                    df.at[idx, 'Notes'] = ''
                    df.at[idx, 'Did the model Predicte the Frame'] = ''

        # Calculate metrics
        if predictions and ground_truths:
            accuracy = accuracy_score(ground_truths, predictions)
            precision = precision_score(ground_truths, predictions, pos_label='yes')
            recall = recall_score(ground_truths, predictions, pos_label='yes')
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"\nModel Metrics:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print(f"F1 Score: {f1:.2%}")

            # Add metrics to the dataframe as a new row
            metrics_row = pd.DataFrame([{
                'ID': 'METRICS',
                'Sentence': '',
                'Frame #': '',
                'Verb': '',
                'Object': '',
                'Envoked Frame': '',
                'Ground Truth (Manually Checking)': '',
                'Frame Definition': '',
                'Notes': f'Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}',
                'Did the model Predicte the Frame': ''
            }])

            df = pd.concat([df, metrics_row], ignore_index=True)

        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")


def main():
    input_csv = "gold_labeling.csv"

    # Microsoft model with advanced method
    output_csv_ms_adv = "results_microsoft_advanced.csv"
    analyzer_ms_adv = FrameAnalyzer(model_name="microsoft/deberta-base-mnli", method="advanced")
    analyzer_ms_adv.process_dataset(input_csv, output_csv_ms_adv)

    # Microsoft model with basic method
    output_csv_ms_basic = "results_microsoft_basic.csv"
    analyzer_ms_basic = FrameAnalyzer(model_name="microsoft/deberta-base-mnli", method="basic")
    analyzer_ms_basic.process_dataset(input_csv, output_csv_ms_basic)

    # MoritzLaurer model with advanced method
    output_csv_ml_adv = "results_moritzlaurer_advanced.csv"
    analyzer_ml_adv = FrameAnalyzer(
        model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        method="advanced"
    )
    analyzer_ml_adv.process_dataset(input_csv, output_csv_ml_adv)

    # MoritzLaurer model with basic method
    output_csv_ml_basic = "results_moritzlaurer_basic.csv"
    analyzer_ml_basic = FrameAnalyzer(
        model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        method="basic"
    )
    analyzer_ml_basic.process_dataset(input_csv, output_csv_ml_basic)


if __name__ == "__main__":
    main()