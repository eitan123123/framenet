import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
from typing import Dict, List, Tuple
from openai import OpenAI
import os
from dotenv import load_dotenv


class FrameInspector:
    def __init__(self):
        """Initialize the frame inspector with NLI model and OpenAI client"""
        self.nlp = spacy.load("en_core_web_sm")

        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=api_key)

        # Initialize MoritzLaurer model
        self.model_name = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def get_first_sentence(self, text: str) -> str:
        """Extract the complete first sentence without truncation"""
        text = text.split("(e.g.")[0].split("(e")[0].strip()
        sentences = text.split('.')
        first_sent = sentences[0].strip().lower()
        first_sent = first_sent.split('(')[0].strip()
        if not first_sent.endswith('.'):
            first_sent += '.'
        first_sent = first_sent.replace('_', ' ')
        return first_sent

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

            # Parse the response
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

            # Direct matching of components
            component_keys = ['who', 'verb', 'object_on_which_the_action_is_made']
            for key in component_keys:
                if key in frame_components and key in sentence_components:
                    substitutions[frame_components[key]] = sentence_components[key]

            # Apply substitutions to frame definition
            modified_definition = frame_definition
            for old, new in substitutions.items():
                modified_definition = modified_definition.replace(old.lower(), new.lower())

            return modified_definition

        except Exception as e:
            print(f"Error in match_and_substitute: {str(e)}")
            return frame_definition

    def calculate_nli_score(self, premise: str, hypothesis: str) -> Dict[str, float]:
        """Calculate NLI scores using the MoritzLaurer model"""
        full_hypothesis = f"This is an event where {hypothesis}"

        inputs = self.tokenizer(
            premise,
            full_hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        return {
            'entailment': scores[0][0].item(),
            'neutral': scores[0][1].item(),
            'contradiction': scores[0][2].item()
        }

    def analyze_csv(self, input_file: str, output_file: str):
        """Analyze the CSV and write detailed results to output file"""
        df = pd.read_csv(input_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            current_sentence_id = None
            current_sentence = None
            sentence_components = None

            for idx, row in df.iterrows():
                sentence_id = row['ID']
                sentence = row['Sentence']
                frame_def = row['Frame Definition']
                frame_num = row['Frame #']
                ground_truth = row['Ground Truth (Manually Checking)']

                # Only extract sentence components once per unique sentence
                if sentence_id != current_sentence_id:
                    current_sentence_id = sentence_id
                    current_sentence = sentence
                    sentence_components = self.extract_components_llm(sentence)

                # Get first sentence of frame definition
                frame_first_sent = self.get_first_sentence(frame_def)

                # Extract frame components
                frame_components = self.extract_components_llm(frame_first_sent, is_frame_definition=True)

                # Create substituted version
                substituted_def = self.match_and_substitute(
                    sentence_components, frame_components, frame_first_sent)

                # Calculate NLI scores
                nli_scores = self.calculate_nli_score(sentence, substituted_def)

                # Write detailed analysis
                f.write(f"{'=' * 80}\n")
                f.write(f"Analysis for ID: {sentence_id}, Frame #{frame_num}\n")
                f.write(f"{'=' * 80}\n\n")

                f.write("Original Data:\n")
                f.write(f"Sentence: {sentence}\n")
                f.write(f"Ground Truth: {ground_truth}\n")
                f.write(f"Frame: {row['Envoked Frame']}\n\n")

                f.write("Sentence Analysis:\n")
                f.write("-" * 20 + "\n")
                for k, v in sentence_components.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")

                f.write("Frame Definition Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Original: {frame_first_sent}\n")
                for k, v in frame_components.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")

                f.write("Component Matching Results:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Substituted Definition: {substituted_def}\n\n")

                f.write("NLI Analysis:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Premise: {sentence}\n")
                f.write(f"Hypothesis: This is an event where {substituted_def}\n")
                for score_type, score in nli_scores.items():
                    f.write(f"{score_type}: {score:.4f}\n")

                f.write("\n" + "=" * 80 + "\n\n")


def main():
    try:
        print("Initializing Frame Inspector...")
        inspector = FrameInspector()

        input_file = '/Users/eitan/Desktop/framenet/datasets/nli_inspection.csv'  # Your input file
        output_file = 'inspection_results.txt'

        print(f"\nAnalyzing file:")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")

        inspector.analyze_csv(input_file, output_file)

        print("\nInspection completed successfully!")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == "__main__":
    main()