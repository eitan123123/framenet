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
        """Initialize the frame inspector with NLI models and OpenAI client"""
        self.nlp = spacy.load("en_core_web_sm")

        # Load environment variables
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")

        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=api_key)

        # Initialize NLI models
        self.models = {
            'microsoft': {
                'name': "microsoft/deberta-base-mnli",
                'tokenizer': None,
                'model': None
            },
            'moritzlaurer': {
                'name': "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                'tokenizer': None,
                'model': None
            }
        }
        self.load_models()

    def load_models(self):
        """Load all NLI models"""
        for key, model_info in self.models.items():
            model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
            model_info['model'] = AutoModelForSequenceClassification.from_pretrained(model_info['name'])

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

    def get_nli_scores(self, model_key: str, premise: str, hypothesis: str) -> Dict[str, float]:
        """Get NLI scores from a specific model"""
        # Add "This is an event where" to the hypothesis
        full_hypothesis = f"This is an event where {hypothesis}"

        model_info = self.models[model_key]
        inputs = model_info['tokenizer'](
            premise,
            full_hypothesis,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        with torch.no_grad():
            outputs = model_info['model'](**inputs)
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
            for idx, row in df.iterrows():
                sentence = row['Sentence']
                frame_def = row['Frame Definition']

                # Get first sentence of frame definition
                first_sentence = self.get_first_sentence(frame_def)

                # Step 1: Extract components
                components = self.extract_components_llm(sentence)

                # Step 2: Create merged sentence
                merged = self.create_merged_sentence(components, first_sentence)

                # Write output in requested format
                f.write(f"Original sentence: {sentence}\n")
                f.write("Extracted components:\n")
                for key, value in components.items():
                    f.write(f"{key}: {value}\n")
                f.write(f"\nFrame definition: {first_sentence}\n")
                f.write(f"Merged sentence: {merged}\n\n")
                f.write("NLI Input:\n")
                f.write(f"Premise: {sentence}\n")
                f.write(f"Hypothesis: This is an event where {merged}\n\n")

                # Get and write NLI scores
                for model_key in ['microsoft', 'moritzlaurer']:
                    scores = self.get_nli_scores(model_key, sentence, merged)
                    f.write(f"{model_key.upper()} Model Scores:\n")
                    for score_type, score in scores.items():
                        f.write(f"{score_type.capitalize()}: {score:.4f}\n")
                    f.write("\n")

                f.write(f"{'=' * 80}\n\n")


def main():
    try:
        print("Initializing Frame Inspector...")
        inspector = FrameInspector()

        input_file = '/Users/eitan/Desktop/framenet/datasets/nli_inspection.csv'
        output_file = 'nli_detailed_analysis_llm.txt'

        print(f"\nAnalyzing file:")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}")

        inspector.analyze_csv(input_file, output_file)

        print("\nInspection completed successfully!")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == "__main__":
    main()