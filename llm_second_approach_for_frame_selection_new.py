import spacy
import nltk
from nltk.corpus import framenet as fn
from typing import Tuple, List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
import numpy as np
from openai import OpenAI
import os
from dotenv import load_dotenv

# Global setup
nlp = spacy.load("en_core_web_sm")
MAX_LENGTH = 512


class FrameAnalyzer:
    def __init__(self, model_name="microsoft/deberta-base-mnli"):
        """Initialize the frame analyzer"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.is_microsoft = "microsoft" in model_name.lower()

        # Load environment variables and initialize OpenAI client
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.openai_client = OpenAI(api_key=api_key)

    def get_first_sentence(self, text: str) -> str:
        """Extract the complete first sentence without truncation"""
        # Clean up parenthetical examples
        text = text.split("(e.g.")[0].split("(e")[0].strip()
        sentences = text.split('.')
        first_sent = sentences[0].strip().lower()
        first_sent = first_sent.split('(')[0].strip()
        if not first_sent.endswith('.'):
            first_sent += '.'
        first_sent = first_sent.replace('_', ' ')
        return first_sent

    def extract_components_llm(self, sentence: str) -> Dict[str, str]:
        """Extract sentence components using LLM"""
        try:
            prompt = """Given a sentence, identify the who (subject), verb, and object. Also specify prepositions and their objects if present.

Examples:
Sentence: I spill water from the bottle
Components:
who: I
verb: spill
object: water
preposition: from
preposition_object: the bottle

Sentence: I throw the bottle
Components:
who: I
verb: throw
object: bottle

Now analyze this sentence:
Sentence: {sentence}
Components:""".format(sentence=sentence)

            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a precise sentence analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )

            # Parse the response to create components dictionary
            components_text = response.choices[0].message.content.strip()
            components = {}
            for line in components_text.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    components[key.strip().lower()] = value.strip()

            return components

        except Exception as e:
            print(f"Error in extract_components_llm: {str(e)}")
            return {}

    def create_merged_sentence(self, components: Dict[str, str], frame_definition: str) -> str:
        """Create merged sentence using components and frame definition"""
        try:
            prompt = """Using the extracted components from a sentence, create a new sentence that follows the structure of a frame definition.

Examples:
Components:
who: I
verb: spill
object: water
preposition: from
preposition_object: the bottle

Frame definition: An Agent or a Cause causes a Fluid to move from a Source to a Goal along a Path or within an Area.
Merged sentence: I spill water from a bottle to a goal

Components:
who: I
verb: throw
object: bottle

Frame definition: A Wearer removes an item of Clothing from a Body location
Merged sentence: I throw the bottle as clothing from a body location

Now merge these:
Components:
{components}

Frame definition: {frame_definition}
Merged sentence:""".format(
                components="\n".join(f"{k}: {v}" for k, v in components.items()),
                frame_definition=frame_definition
            )

            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": "You are a precise sentence merger."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=100
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error in create_merged_sentence: {str(e)}")
            return f"Error in creating merged sentence: {str(e)}"

    def calculate_semantic_similarity(self, premise: str, hypothesis: str) -> float:
        """Calculate semantic similarity based on model type"""
        # Add "This is an event where" to the hypothesis
        full_hypothesis = f"This is an event where {hypothesis}"

        inputs = self.tokenizer(premise, full_hypothesis,
                                return_tensors="pt",
                                padding=True,
                                truncation=True,
                                max_length=MAX_LENGTH)

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        if self.is_microsoft:
            return scores[0][0].item()  # Entailment score
        else:
            return scores[0][0].item()  # Entailment score

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
        print(f"\nProcessing dataset: {input_csv}")
        df = pd.read_csv(input_csv)

        # Initialize columns
        new_columns = ['Frame Definition', 'Notes', 'Did the model Predicte the Frame',
                       'LLM Merged Sentence', 'Sentence Components']
        for col in new_columns:
            if col not in df.columns:
                df[col] = ''

        # Convert columns to string type
        string_columns = ['Envoked Frame', 'Ground Truth (Manually Checking)',
                          'Frame Definition', 'Notes', 'Did the model Predicte the Frame',
                          'LLM Merged Sentence', 'Sentence Components']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype('string')

        results_dict = {}
        predictions = []
        ground_truths = []

        # Process unique sentences
        unique_sentences = df.drop_duplicates(subset=['ID', 'Sentence'])

        print("\nProcessing unique sentences...")
        for idx, row in unique_sentences.iterrows():
            print(f"Processing sentence {idx + 1}/{len(unique_sentences)}")
            sentence_id = row['ID']
            sentence = row['Sentence']
            verb = row['Verb']

            if sentence_id not in results_dict:
                frames = self.find_relevant_frames(verb)
                results = []

                # First step: Extract components
                components = self.extract_components_llm(sentence)
                components_str = "\n".join(f"{k}: {v}" for k, v in components.items())

                for frame_idx, frame in enumerate(frames, 1):
                    print(f"Processing frame {frame_idx}/{len(frames)} for sentence {idx + 1}")

                    # Get first sentence of frame definition
                    frame_first_sent = self.get_first_sentence(frame["definition"])

                    # Second step: Create merged sentence
                    merged = self.create_merged_sentence(components, frame_first_sent)

                    # Calculate similarity
                    similarity = self.calculate_semantic_similarity(sentence, merged)

                    results.append({
                        "frame": frame["name"],
                        "definition": frame["definition"],
                        "similarity_score": similarity,
                        "llm_merged": merged,
                        "components": components_str
                    })

                results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
                results_dict[sentence_id] = results

        # Update DataFrame and collect metrics
        print("\nUpdating results in DataFrame...")
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
                    df.at[idx, 'LLM Merged Sentence'] = frame_result['llm_merged']
                    df.at[idx, 'Sentence Components'] = frame_result['components']
                    df.at[idx, 'Notes'] = f"Confidence: {confidence:.3f}"

                    prediction = 'yes' if confidence >= 0.6 else 'no'
                    df.at[idx, 'Did the model Predicte the Frame'] = 'v' if prediction == 'yes' else ''

                    if ground_truth and ground_truth.strip().lower() in ['yes', 'no']:
                        predictions.append(prediction)
                        ground_truths.append(ground_truth.strip().lower())

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
                'Did the model Predicte the Frame': '',
                'LLM Merged Sentence': '',
                'Sentence Components': ''
            }])

            df = pd.concat([metrics_row, df], ignore_index=True)

        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")


def main():
    try:
        print("Initializing Frame Analyzer...")

        # Microsoft model
        output_csv_ms = "results_microsoft_llm.csv"
        analyzer_ms = FrameAnalyzer(model_name="microsoft/deberta-base-mnli")
        analyzer_ms.process_dataset("/Users/eitan/Desktop/framenet/datasets/gold_labeling.csv", output_csv_ms)

        # MoritzLaurer model
        output_csv_ml = "results_moritzlaurer_llm.csv"
        analyzer_ml = FrameAnalyzer(
            model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        )
        analyzer_ml.process_dataset("/Users/eitan/Desktop/framenet/datasets/gold_labeling.csv", output_csv_ml)

        print("\nAnalysis completed successfully!")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == "__main__":
    main()