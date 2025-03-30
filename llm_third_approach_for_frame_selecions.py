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

    def extract_components_llm(self, text: str, is_frame_definition: bool = False) -> Dict[str, str]:
        """Extract specific sentence components using LLM"""
        try:
            if is_frame_definition:
                prompt = """Given a frame definition sentence, identify exactly these components:
    who: (the agent/person/entity performing the action)
    verb: (the main action being performed)
    object_on_which_the_action_is_made: (what receives the action)

    Examples:
    Frame definition: An Agent manipulates a Fastener to open or close a Containing_object
    Components:
    who: Agent
    verb: manipulates
    object_on_which_the_action_is_made: Fastener

    Frame definition: A Victim undergoes injury from contact with an Injuring_entity
    Components:
    who: Victim
    verb: undergoes
    object_on_which_the_action_is_made: injury

    Now analyze this frame definition, providing ONLY these three components:
    Frame definition: {text}
    Components:""".format(text=text)
            else:
                prompt = """Given a sentence, identify exactly these components:
    who: (the agent/person/entity performing the action)
    verb: (the main action being performed)
    object_on_which_the_action_is_made: (what receives the action)

    Examples:
    Sentence: I spill water from the bottle
    Components:
    who: I
    verb: spill
    object_on_which_the_action_is_made: water

    Sentence: The cat scratches the door
    Components:
    who: cat
    verb: scratches
    object_on_which_the_action_is_made: door

    Now analyze this sentence, providing ONLY these three components:
    Sentence: {text}
    Components:""".format(text=text)

            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system",
                     "content": "You are a precise sentence analyzer. Only output the requested components."},
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

    def match_and_substitute(self, sentence_components: Dict[str, str],
                             frame_components: Dict[str, str],
                             frame_definition: str) -> str:
        """Match and substitute components with exact matching"""
        try:
            substitutions = {}

            # Direct matching of who/verb/object components
            if ('who' in frame_components and 'who' in sentence_components):
                substitutions[frame_components['who']] = sentence_components['who']

            if ('verb' in frame_components and 'verb' in sentence_components):
                substitutions[frame_components['verb']] = sentence_components['verb']

            if ('object_on_which_the_action_is_made' in frame_components and
                    'object_on_which_the_action_is_made' in sentence_components):
                substitutions[frame_components['object_on_which_the_action_is_made']] = \
                    sentence_components['object_on_which_the_action_is_made']

            # Apply substitutions to frame definition
            modified_definition = frame_definition
            for old, new in substitutions.items():
                modified_definition = modified_definition.replace(old.lower(), new.lower())

            return modified_definition

        except Exception as e:
            print(f"Error in match_and_substitute: {str(e)}")
            return frame_definition

    def verify_dataset(self, df: pd.DataFrame):
        """Verify the dataset structure and values"""
        print("\n=== Dataset Verification ===")

        # Check for duplicate IDs with conflicting ground truths
        for id_val in df['ID'].unique():
            id_data = df[df['ID'] == id_val]
            unique_ground_truths = id_data['Ground Truth (Manually Checking)'].unique()
            if len(unique_ground_truths) > 1:
                print(f"WARNING: ID {id_val} has multiple ground truth values: {unique_ground_truths}")
                print(id_data[['Frame #', 'Envoked Frame', 'Ground Truth (Manually Checking)']])

        # Verify frame numbers are sequential for each ID
        for id_val in df['ID'].unique():
            id_data = df[df['ID'] == id_val]
            frame_nums = id_data['Frame #'].tolist()
            expected_nums = list(range(1, len(frame_nums) + 1))
            if frame_nums != expected_nums:
                print(f"WARNING: Non-sequential frame numbers for ID {id_val}")
                print(f"Expected: {expected_nums}")
                print(f"Found: {frame_nums}")

        # Check Experience_bodily_harm frames specifically
        bodily_harm = df[df['Envoked Frame'] == 'Experience_bodily_harm']
        print("\nExperience_bodily_harm frames:")
        print(bodily_harm[['ID', 'Frame #', 'Ground Truth (Manually Checking)']])



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

    def get_first_sentence(self, text: str) -> str:
        """Extract the complete first sentence of frame definition"""
        text = text.split("(e.g.")[0].split("(e")[0].strip()
        sentences = text.split('.')
        first_sent = sentences[0].strip().lower()
        first_sent = first_sent.split('(')[0].strip()
        if not first_sent.endswith('.'):
            first_sent += '.'
        first_sent = first_sent.replace('_', ' ')
        return first_sent

    def process_dataset(self, input_csv: str, output_csv: str):
        """Process dataset with component matching and evaluation metrics"""
        print(f"\nProcessing dataset: {input_csv}")
        df = pd.read_csv(input_csv)

        # Debug print initial data for ID 8
        print("\n=== Initial Data for ID 8 ===")
        id_8_data = df[df['ID'] == 8]
        print(id_8_data[['ID', 'Frame #', 'Sentence', 'Envoked Frame', 'Ground Truth (Manually Checking)']])

        # Initialize columns
        new_columns = ['Frame Definition', 'Notes', 'Did the model Predicte the Frame',
                       'LLM Merged Sentence', 'Sentence Components', 'Frame Components']
        for col in new_columns:
            if col not in df.columns:
                df[col] = ''

        # Convert columns to string type
        string_columns = ['Envoked Frame', 'Ground Truth (Manually Checking)',
                          'Frame Definition', 'Notes', 'Did the model Predicte the Frame',
                          'LLM Merged Sentence', 'Sentence Components', 'Frame Components']
        for col in string_columns:
            if col in df.columns:
                df[col] = df[col].astype('string')

        # Store original values
        original_data = df[['ID', 'Frame #', 'Envoked Frame', 'Ground Truth (Manually Checking)']].copy()

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

                # Extract sentence components
                sentence_components = self.extract_components_llm(sentence)
                components_str = "\n".join(f"{k}: {v}" for k, v in sentence_components.items())

                for frame_idx, frame in enumerate(frames, 1):
                    print(f"Processing frame {frame_idx}/{len(frames)} for sentence {idx + 1}")

                    # Get first sentence of frame definition
                    frame_first_sent = self.get_first_sentence(frame["definition"])

                    # Extract frame definition components
                    frame_components = self.extract_components_llm(frame_first_sent, is_frame_definition=True)
                    frame_components_str = "\n".join(f"{k}: {v}" for k, v in frame_components.items())

                    # Create substituted version
                    substituted_definition = self.match_and_substitute(
                        sentence_components, frame_components, frame_first_sent)

                    # Calculate similarity
                    similarity = self.calculate_semantic_similarity(sentence, substituted_definition)

                    results.append({
                        "frame": frame["name"],
                        "definition": frame["definition"],
                        "similarity_score": similarity,
                        "llm_merged": substituted_definition,
                        "sentence_components": components_str,
                        "frame_components": frame_components_str
                    })

                results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
                results_dict[sentence_id] = results

        # Debug print before DataFrame update for ID 8
        print("\n=== Before DataFrame Update (ID 8) ===")
        if 8 in results_dict:
            print("Original data for ID 8:")
            print(original_data[original_data['ID'] == 8])

        # Update DataFrame and collect metrics
        print("\nUpdating results in DataFrame...")
        for idx, row in df.iterrows():
            sentence_id = row['ID']
            frame_num = row['Frame #']

            if sentence_id in results_dict:
                results = results_dict[sentence_id][:12]  # Take top 12 frames

                if frame_num <= len(results):
                    frame_result = results[frame_num - 1]
                    confidence = frame_result['similarity_score']

                    # Debug print during update for ID 8
                    if sentence_id == 8:
                        print(f"\nProcessing ID 8, Frame #{frame_num}")
                        print(f"Original frame: {original_data.loc[idx, 'Envoked Frame']}")
                        print(f"Original ground truth: {original_data.loc[idx, 'Ground Truth (Manually Checking)']}")

                    # Restore original values from stored data
                    df.at[idx, 'Envoked Frame'] = original_data.loc[idx, 'Envoked Frame']
                    df.at[idx, 'Ground Truth (Manually Checking)'] = original_data.loc[
                        idx, 'Ground Truth (Manually Checking)']

                    # Update only additional columns
                    df.at[idx, 'Frame Definition'] = frame_result['definition']
                    df.at[idx, 'LLM Merged Sentence'] = frame_result['llm_merged']
                    df.at[idx, 'Sentence Components'] = frame_result['sentence_components']
                    df.at[idx, 'Frame Components'] = frame_result['frame_components']
                    df.at[idx, 'Notes'] = f"Confidence: {confidence:.3f}"

                    # Calculate prediction but use original ground truth for metrics
                    prediction = 'yes' if confidence >= 0.6 else 'no'
                    df.at[idx, 'Did the model Predicte the Frame'] = 'v' if prediction == 'yes' else ''

                    ground_truth = original_data.loc[idx, 'Ground Truth (Manually Checking)']
                    if ground_truth and ground_truth.strip().lower() in ['yes', 'no']:
                        predictions.append(prediction)
                        ground_truths.append(ground_truth.strip().lower())

                    # Debug print after update for ID 8
                    if sentence_id == 8:
                        print(f"After update for ID 8:")
                        print(f"Frame: {df.at[idx, 'Envoked Frame']}")
                        print(f"Ground Truth: {df.at[idx, 'Ground Truth (Manually Checking)']}")

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
                'Sentence Components': '',
                'Frame Components': ''
            }])

            df = pd.concat([metrics_row, df], ignore_index=True)

        # Debug print final state for ID 8
        print("\n=== Final State for ID 8 ===")
        final_id_8_data = df[df['ID'] == 8]
        print(final_id_8_data[['ID', 'Frame #', 'Sentence', 'Envoked Frame', 'Ground Truth (Manually Checking)']])

        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")


import os


def ensure_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = "frame_analysis_results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def run_analysis():
    """Run the frame analysis with both models"""
    try:
        output_dir = ensure_output_directory()
        input_file = "/Users/eitan/Desktop/framenet/datasets/gold_labeling.csv"  # Your input file

        # Run Microsoft model
        print("\n=== Running Microsoft DeBERTa Model ===")
        output_file_ms = os.path.join(output_dir, "results_microsoft_llm.csv")
        analyzer_ms = FrameAnalyzer(model_name="microsoft/deberta-base-mnli")
        analyzer_ms.process_dataset(input_file, output_file_ms)  # Removed preserve_gold_labels parameter

        # Run MoritzLaurer model
        print("\n=== Running MoritzLaurer Model ===")
        output_file_ml = os.path.join(output_dir, "results_moritzlaurer_llm.csv")
        analyzer_ml = FrameAnalyzer(
            model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        )
        analyzer_ml.process_dataset(input_file, output_file_ml)  # Removed preserve_gold_labels parameter

        print("\nAnalysis completed!")
        print(f"Results saved in {output_dir}/")

    except Exception as e:
        print(f"\nError in analysis: {str(e)}")


if __name__ == "__main__":
    run_analysis()