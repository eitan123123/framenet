import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import os
from openai import OpenAI
from sklearn.metrics import accuracy_score, recall_score, precision_score
from dotenv import load_dotenv
from tqdm import tqdm
import re


class EnhancedFrameAnalyzer:
    def __init__(self, model_name="microsoft/deberta-base-mnli"):
        """Initialize the frame analyzer with NLI model and OpenAI client"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.is_microsoft = "microsoft" in model_name.lower()

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.openai_client = OpenAI(api_key=api_key)

        # Threshold for frame acceptance
        self.ACCEPTANCE_THRESHOLD = 0.6

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

    def extract_core_elements(self, frame_definition: str) -> List[Dict[str, str]]:
        """Extract core elements from FrameNet definition"""
        try:
            # Get original frame definition and first sentence
            first_sentence = self.get_first_sentence(frame_definition)
            original_sentence = frame_definition.split('.')[0].strip()

            # Find terms that are likely to be core elements (looking in original to preserve case)
            core_elements = []

            # Use regex to find capitalized terms that might be core elements
            # This pattern looks for:
            # 1. Words starting with capital letters
            # 2. Multi-word terms with underscores
            # 3. Preserves original capitalization
            pattern = re.compile(r'\b[A-Z][a-zA-Z]*(?:_[A-Z][a-zA-Z]*)*\b')

            matches = pattern.finditer(original_sentence)
            seen_elements = set()

            for match in matches:
                element = match.group()
                # Clean up and validate the element
                if (element not in ['A', 'An', 'The', 'This', 'That', 'These', 'Those'] and
                        element not in seen_elements):
                    # Remove any trailing punctuation
                    element = element.rstrip('.,;:()')
                    if element:
                        seen_elements.add(element)
                        # Get surrounding context (10 words before and after)
                        start_idx = max(0, match.start() - 50)
                        end_idx = min(len(original_sentence), match.end() + 50)
                        context = original_sentence[start_idx:end_idx].strip()

                        core_elements.append({
                            'name': element,
                            'context': context
                        })

            print(f"Found core elements: {[elem['name'] for elem in core_elements]}")
            return core_elements

        except Exception as e:
            print(f"Error in extract_core_elements: {str(e)}")
            return []

    def map_elements_to_sentence(self, sentence: str, core_elements: List[Dict[str, str]], frame_definition: str) -> \
    Dict[str, str]:
        """Map FrameNet core elements to parts of the input sentence using LLM"""
        mappings = {}

        for element in core_elements:
            try:
                prompt = (
                    f"As a FrameNet analyzer, examine this sentence and find what part (if any) "
                    f"corresponds to the given semantic role (core element).\n\n"
                    f"Input Sentence: {sentence}\n"
                    f"Core Element: {element['name']}\n"
                    f"Element Context: {element['context']}\n\n"
                    f"Rules:\n"
                    f"1. Return ONLY the exact word or phrase from the input sentence that plays this semantic role\n"
                    f"2. If no word/phrase corresponds to this role, return exactly \"None\"\n"
                    f"3. Do not explain or add any other text\n\n"
                    f"Example format:\n"
                    f"Input: \"John broke the window\"\n"
                    f"Core Element: \"Agent\"\n"
                    f"Response: John\n\n"
                    f"Input: \"The window broke\"\n"
                    f"Core Element: \"Agent\"\n"
                    f"Response: None\n\n"
                    f"Your response for this case:"
                )

                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {"role": "system", "content": "You are a precise semantic role labeler for FrameNet analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1
                )

                mapped_text = response.choices[0].message.content.strip()
                mappings[element['name']] = mapped_text if mapped_text.lower() != "none" else None

            except Exception as e:
                print(f"Error mapping element {element['name']}: {str(e)}")
                mappings[element['name']] = None

        return mappings

    def create_nli_hypothesis(self, frame_first_sentence: str, element_mappings: Dict[str, str]) -> str:
        """Create NLI hypothesis by substituting mapped elements"""
        hypothesis = frame_first_sentence

        # Replace mapped elements in the hypothesis
        for element_name, mapped_text in element_mappings.items():
            if mapped_text:
                hypothesis = re.sub(r'\b' + re.escape(element_name) + r'\b', mapped_text, hypothesis,
                                    flags=re.IGNORECASE)

        return f"This describes a situation in which {hypothesis}"

    def calculate_semantic_similarity(self, premise: str, hypothesis: str) -> float:
        """Calculate semantic similarity based on model type"""
        try:
            inputs = self.tokenizer(premise, hypothesis,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True,
                                    max_length=512)

            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)

            if self.is_microsoft:
                return scores[0][0].item()  # Entailment score for Microsoft model
            else:
                return scores[0][0].item()  # Entailment score for MoritzLaurer model

        except Exception as e:
            print(f"Error in calculate_semantic_similarity: {str(e)}")
            return 0.0

    def process_dataset(self, input_csv: str, output_csv: str):
        """Process the dataset and generate analysis results"""
        try:
            # Read dataset
            df = pd.read_csv(input_csv)

            # Initialize new columns
            new_columns = [
                'First_Sentence', 'Core_Elements', 'Element_Mappings',
                'NLI_Hypothesis', 'Entailment_Score', 'Frame_Accepted',
                'Ground_Truth'
            ]

            for col in new_columns:
                if col not in df.columns:
                    df[col] = None

            results = []
            predictions = []
            ground_truths = []

            # Process each row with progress bar
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing rows", position=0, leave=True):
                # Get first sentence from frame definition
                first_sentence = self.get_first_sentence(row['Frame Definition'])

                # Extract core elements from FrameNet definition
                core_elements = self.extract_core_elements(row['Frame Definition'])

                # Map elements to original sentence
                element_mappings = self.map_elements_to_sentence(
                    row['Sentence'],
                    core_elements,
                    first_sentence
                )

                # Create NLI hypothesis
                hypothesis = self.create_nli_hypothesis(first_sentence, element_mappings)

                # Calculate entailment score
                score = self.calculate_semantic_similarity(row['Sentence'], hypothesis)

                # Determine if frame is accepted
                frame_accepted = score >= self.ACCEPTANCE_THRESHOLD

                # Store results with proper dictionary formatting
                element_mappings_dict = {}
                for element in core_elements:
                    mapped_text = self.map_elements_to_sentence(
                        row['Sentence'],
                        [element],  # Pass single element as list
                        first_sentence
                    ).get(element['name'])
                    if mapped_text and mapped_text.lower() != 'none':
                        element_mappings_dict[element['name']] = mapped_text

                results.append({
                    'Sentence': row['Sentence'],
                    'Verb': row['Verb'],
                    'Frame_Name': row['Envoked Frame'],  # Add Frame Name
                    'Frame_Definition': row['Frame Definition'],
                    'First_Sentence': first_sentence,
                    'Core_Elements_Mappings': element_mappings_dict,  # Store as dictionary
                    'NLI_Hypothesis': hypothesis,
                    'Entailment_Score': score,
                    'Frame_Accepted': frame_accepted,
                    'Ground_Truth': row['Ground Truth (Manually Checking)'] == 'yes'
                })

                # Collect metrics data
                if row['Ground Truth (Manually Checking)']:
                    predictions.append('yes' if frame_accepted else 'no')
                    ground_truths.append(row['Ground Truth (Manually Checking)'].strip().lower())

            # Create results DataFrame
            results_df = pd.DataFrame(results)

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

                # Add metrics to DataFrame
                metrics_row = pd.DataFrame([{
                    'Sentence': 'METRICS',
                    'First_Sentence': '',
                    'Core_Elements': '',
                    'Element_Mappings': '',
                    'NLI_Hypothesis': '',
                    'Entailment_Score': '',
                    'Frame_Accepted': '',
                    'Ground_Truth': '',
                    'Notes': f'Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}'
                }])

                results_df = pd.concat([metrics_row, results_df], ignore_index=True)

            # Save results
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")

        except Exception as e:
            print(f"Error in process_dataset: {str(e)}")


def main():
    try:
        print("Initializing Frame Analyzer...")
        load_dotenv()

        input_file = "/Users/eitan/Desktop/framenet/datasets/gold_labeling.csv"

        # Microsoft model
        print("\n=== Running Microsoft DeBERTa Model ===")
        output_csv_ms = "results_microsoft_frames.csv"
        analyzer_ms = EnhancedFrameAnalyzer(model_name="microsoft/deberta-base-mnli")
        analyzer_ms.process_dataset(input_file, output_csv_ms)

        # MoritzLaurer model
        print("\n=== Running MoritzLaurer Model ===")
        output_csv_ml = "results_moritzlaurer_frames.csv"
        analyzer_ml = EnhancedFrameAnalyzer(
            model_name="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        )
        analyzer_ml.process_dataset(input_file, output_csv_ml)

        print("\nAnalysis completed!")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == "__main__":
    main()