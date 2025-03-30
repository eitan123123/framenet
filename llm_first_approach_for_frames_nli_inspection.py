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

        # Get first complete sentence
        sentences = text.split('.')
        first_sent = sentences[0].strip().lower()

        # Remove dangling parentheses
        first_sent = first_sent.split('(')[0].strip()

        # Add period if not present
        if not first_sent.endswith('.'):
            first_sent += '.'

        # Remove underscores
        first_sent = first_sent.replace('_', ' ')

        return first_sent

    def extract_sentence_components(self, sentence: str) -> Dict[str, str]:
        """Extract subject, verb, and object from sentence"""
        doc = self.nlp(sentence)
        subject = next((token for token in doc if token.dep_ == "nsubj"), None)
        verb = next((token for token in doc if token.pos_ == "VERB"), None)
        obj = next((token for token in doc if token.dep_ == "dobj"), None)

        return {
            "subject": subject.text if subject else "",
            "verb": verb.text if verb else "",
            "object": obj.text if obj else ""
        }

    def create_llm_hypothesis(self, sentence: str, frame_definition: str) -> str:
        """Create a merged sentence using LLM"""
        try:
            components = self.extract_sentence_components(sentence)

            prompt = """You are tasked with merging a premise sentence with a frame definition to create a new sentence.
The goal is to take the specific elements from the premise (subject, verb, object) and integrate them into the frame structure.
IMPORTANT: ALWAYS keep the original verb from the premise, never replace it with verbs from the frame definition.

Examples:

Premise: I spill water from the bottle
Frame definition: An Agent or a Cause causes a Fluid to move from a Source to a Goal along a Path or within an Area.
Merged sentence: I spill water from a bottle to a goal

and sometimes it doesnt make sense in the sentense and i still and you to give me the different injected parts of the sentense in the frame defenition
Premise: I throw the bottle
Frame definition: A Wearer removes an item of Clothing from a Body location
Merged sentence: I throw the bottle as clothing from a body location

For exact merging:
1. ALWAYS keep the original verb - never replace it with frame definition verbs
2. Keep the original subject
3. Keep the original object
4. Add frame elements around these preserved components
5. Use the frame elements from the frame definition and map them to the premise components

Now merge:
Premise: {premise}
Frame definition: {frame_definition}

Using these components from premise:
Subject: {subject}
Verb: {verb}
Object: {object}

Merged sentence:""".format(
                premise=sentence,
                frame_definition=frame_definition,
                subject=components['subject'],
                verb=components['verb'],
                object=components['object']
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
            print(f"Error in create_llm_hypothesis: {str(e)}")
            return f"Error in generating hypothesis: {str(e)}"

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

                # Create merged sentence
                merged = self.create_llm_hypothesis(sentence, first_sentence)

                # Write output in requested format
                f.write(f"Original sentence: {sentence}\n")
                f.write(f"Frame definition: {first_sentence}\n")
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

        input_file = 'nli_inspection.csv'
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