import spacy
import nltk
from nltk.corpus import framenet as fn
from typing import Tuple, List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# ================================
#       GLOBAL SETUP
# ================================
nlp = spacy.load("en_core_web_sm")

# Use the microsoft/deberta-base-mnli model (instead of MoritzLaurer)

#"MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
tokenizer = AutoTokenizer.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
model = AutoModelForSequenceClassification.from_pretrained("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")

# You can choose a different MAX_LENGTH if desired
MAX_LENGTH = 512


# ================================
#   HELPER FUNCTIONS
# ================================

def find_relevant_frames(verb: str) -> List[Dict[str, str]]:
    """
    Find frames relevant to the given verb (searching in FrameNet).
    Same as in the first code, just returns frame name + definition.
    """
    frames = []
    verb_search = verb.lower()

    for frame in fn.frames():
        for lu in frame.lexUnit.keys():
            # We only look for lexical units of type .v that contain our verb
            if verb_search in lu.lower() and '.v' in lu.lower():
                frame_info = {
                    "name": frame.name,
                    "definition": frame.definition
                }
                frames.append(frame_info)
                break

    return frames


def create_natural_hypothesis(sentence: str, frame: Dict[str, str]) -> str:
    """
    Instead of the multiple-sentence approach from the first code,
    we just create a single hypothesis that concatenates:
      "This is an event where {first line of frame_def}"
    as in your second code snippet.
    """
    # Take the frame definition up to the first period or the whole thing if no period
    frame_def = frame["definition"].split('.')[0].strip().lower()

    # Return the simple concatenation
    return f"This is an event where {frame_def}"







# the other model that is not the microsoft one, the next 2 functions
def calculate_nli_entailment_score(premise: str, hypothesis: str) -> float:
    """
    Calculate the entailment score using DeBERTa NLI model.
    Returns the entailment probability specifically.
    """
    inputs = tokenizer(premise, hypothesis,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Index 0 corresponds to entailment in this model
    return scores[0][0].item()

def calculate_semantic_similarity(premise: str, hypothesis: str) -> float:
    """
    Calculate semantic similarity using entailment score from NLI model.
    This function now returns the entailment probability specifically.
    """
    inputs = tokenizer(premise, hypothesis,
                       return_tensors="pt",
                       padding=True,
                       truncation=True,
                       max_length=MAX_LENGTH)

    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)

    # Return entailment score (index 0) instead of contradiction score (index 2)
    return scores[0][0].item()






#microsoft model
# def calculate_semantic_similarity(premise: str, hypothesis: str) -> float:
#     """Original calculate_semantic_similarity function"""
#     inputs = tokenizer(premise, hypothesis,
#                       return_tensors="pt",
#                       padding=True,
#                       truncation=True,
#                       max_length=MAX_LENGTH)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#         scores = torch.nn.functional.softmax(outputs.logits, dim=1)
#
#     return scores[0][2].item()











# ================================
#   CSV-BASED EVALUATION
# ================================

def process_dataset_with_evaluation(input_csv: str, output_csv: str):
    """
    Process the dataset, add frame predictions and compare with ground truth.
    We keep exactly the same logic as in your first code, but now the model
    and hypothesis creation are from the 'microsoft/deberta-base-mnli' approach.
    """
    # Read input CSV
    df = pd.read_csv(input_csv)

    # Initialize new columns if they don't exist
    if 'Frame Definition' not in df.columns:
        df['Frame Definition'] = ''
    if 'Notes' not in df.columns:
        df['Notes'] = ''
    if 'Did the model Predicte the Frame' not in df.columns:
        df['Did the model Predicte the Frame'] = ''

    # Explicitly set column types to string where needed
    string_columns = [
        'Envoked Frame',
        'Ground Truth (Manually Checking)',
        'Frame Definition',
        'Notes',
        'Did the model Predicte the Frame'
    ]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype('string')

    # We'll store the results for each (ID,Sentence) pair so we only compute them once
    results_dict = {}
    total_predictions = 0
    correct_predictions = 0

    # Identify the unique rows to avoid duplicate processing
    unique_sentences = df.drop_duplicates(subset=['ID', 'Sentence'])

    # 1) Collect frames & scores
    for _, row in unique_sentences.iterrows():
        sentence_id = row['ID']
        sentence = row['Sentence']
        verb = row['Verb']

        if sentence_id not in results_dict:
            # For each unique sentence, find all relevant frames
            frames = find_relevant_frames(verb)
            results = []

            for frame in frames:
                hypothesis = create_natural_hypothesis(sentence, frame)
                similarity = calculate_semantic_similarity(sentence, hypothesis)

                result = {
                    "frame": frame["name"],
                    "definition": frame["definition"],
                    "hypothesis": hypothesis,
                    "similarity_score": similarity
                }
                results.append(result)

            # Sort by similarity score in descending order
            results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
            results_dict[sentence_id] = results

    # 2) Assign frames to rows in the CSV + Evaluate
    for idx, row in df.iterrows():
        sentence_id = row['ID']
        frame_num = row['Frame #']
        ground_truth = row['Ground Truth (Manually Checking)']

        if sentence_id in results_dict:
            results = results_dict[sentence_id]

            # If more than 12 frames found, truncate to top 12
            if len(results) > 12:
                results = results[:12]

            # If we have enough frames for this row's 'Frame #'
            if frame_num <= len(results):
                frame_result = results[frame_num - 1]
                confidence = frame_result['similarity_score']

                # Fill columns in the dataframe
                df.at[idx, 'Envoked Frame'] = frame_result['frame']
                df.at[idx, 'Frame Definition'] = frame_result['definition']
                df.at[idx, 'Notes'] = f"Confidence: {confidence:.3f}"

                # The first code uses a threshold of 0.7 for "Did the model Predict"
                prediction = 'v' if confidence >= 0.7 else ''
                df.at[idx, 'Did the model Predicte the Frame'] = prediction

                # Compare to ground truth for an approximate accuracy measure
                if ground_truth and ground_truth.strip().lower() == 'yes':
                    total_predictions += 1
                    if confidence >= 0.7:
                        correct_predictions += 1
                elif ground_truth and ground_truth.strip().lower() == 'no':
                    total_predictions += 1
                    if confidence < 0.7:
                        correct_predictions += 1
            else:
                # Clear fields if no suitable frame is found
                df.at[idx, 'Envoked Frame'] = ''
                df.at[idx, 'Frame Definition'] = ''
                df.at[idx, 'Notes'] = ''
                df.at[idx, 'Did the model Predicte the Frame'] = ''

    # 3) Final accuracy
    accuracy = (correct_predictions / total_predictions * 100) if total_predictions > 0 else 0
    print(f"Model Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_predictions} correct predictions)")

    # 4) Save updated DataFrame
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def process_dataset_with_definitions(input_csv: str, output_csv: str):
    """
    Similar to the above, but doesn't evaluate correctness with ground truth.
    Instead, it just writes the 'Frame Definition' + 'Predicted Frame' columns.
    This also matches your first code structure, but simplified.
    """
    df = pd.read_csv(input_csv)

    # Add new column for frame definitions if they don't exist
    if 'Frame Definition' not in df.columns:
        df['Frame Definition'] = ''

    # Ensure columns are strings
    string_columns = ['Envoked Frame', 'Predicted Frame', 'Notes', 'Frame Definition']
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].astype('string')

    # Cache the results for each (ID, Sentence)
    results_dict = {}
    unique_sentences = df.drop_duplicates(subset=['ID', 'Sentence'])

    for _, row in unique_sentences.iterrows():
        sentence_id = row['ID']
        sentence = row['Sentence']
        verb = row['Verb']  # or however your CSV lists it

        if sentence_id not in results_dict:
            frames = find_relevant_frames(verb)
            results = []
            for frame in frames:
                hypothesis = create_natural_hypothesis(sentence, frame)
                similarity = calculate_semantic_similarity(sentence, hypothesis)
                results.append({
                    "frame": frame["name"],
                    "definition": frame["definition"],
                    "hypothesis": hypothesis,
                    "similarity_score": similarity
                })

            # Sort descending by similarity
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            results_dict[sentence_id] = results

    # Write out the results to CSV
    for idx, row in df.iterrows():
        sentence_id = row['ID']
        frame_num = row['Frame #']

        if sentence_id in results_dict:
            results = results_dict[sentence_id]
            if len(results) > 12:
                results = results[:12]

            if frame_num <= len(results):
                frame_result = results[frame_num - 1]
                df.at[idx, 'Envoked Frame'] = frame_result['frame']
                df.at[idx, 'Frame Definition'] = frame_result['definition']
                df.at[idx, 'Predicted Frame'] = 'v' if frame_result['similarity_score'] >= 0.7 else ''
                df.at[idx, 'Notes'] = f"Confidence: {frame_result['similarity_score']:.3f}"
            else:
                # Clear fields if no suitable frame
                df.at[idx, 'Envoked Frame'] = ''
                df.at[idx, 'Frame Definition'] = ''
                df.at[idx, 'Predicted Frame'] = ''
                df.at[idx, 'Notes'] = ''

    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


def analyze_sentence_with_frames(sentence: str, threshold: float = 0.0) -> List[Dict]:
    """
    If you still want a direct function (like in the first code)
    that returns all frames for a single sentence, you can keep this.
    By default threshold=0.0 to return everything. Or change as desired.
    """
    doc = nlp(sentence)
    verb = next((token.text for token in doc if token.pos_ == "VERB"), None)

    if not verb:
        return []

    frames = find_relevant_frames(verb)
    if not frames:
        return []

    results = []
    for frame in frames:
        hypothesis = create_natural_hypothesis(sentence, frame)
        similarity = calculate_semantic_similarity(sentence, hypothesis)
        results.append({
            "frame": frame["name"],
            "definition": frame["definition"],
            "hypothesis": hypothesis,
            "similarity_score": similarity
        })

    results = sorted(results, key=lambda x: x["similarity_score"], reverse=True)
    # If threshold>0, you could filter:
    # results = [r for r in results if r["similarity_score"] >= threshold]
    return results


# ================================
#         MAIN BLOCK
# ================================
if __name__ == "__main__":
    # Example usage with your CSVs
    input_csv = "my_dataset_checkpoint_with_my_evaluation.csv"
    output_csv = "output_basic_mathod_results_with_MoritzLaurer_nli_model.csv"

    try:
        process_dataset_with_evaluation(input_csv, output_csv)
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

    # If you want, you can also call:
    # process_dataset_with_definitions(input_csv, "some_other_output.csv")
