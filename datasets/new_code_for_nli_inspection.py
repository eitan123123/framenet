
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import spacy
from typing import Dict, List, Tuple

class NLIAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
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
        for key, model_info in self.models.items():
            model_info['tokenizer'] = AutoTokenizer.from_pretrained(model_info['name'])
            model_info['model'] = AutoModelForSequenceClassification.from_pretrained(model_info['name'])

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

    def create_basic_hypothesis(self, frame_def: str) -> str:
        first_sentence = self.get_first_sentence(frame_def)
        return f"This is an event where {first_sentence}"

    def create_advanced_hypothesis(self, sentence: str, frame_def: str) -> List[str]:
        doc = self.nlp(sentence)
        subject = next((t for t in doc if t.dep_ == "nsubj"), None)
        verb = next((t for t in doc if t.pos_ == "VERB"), None)
        obj = next((t for t in doc if t.dep_ == "dobj"), None)

        first_sentence = self.get_first_sentence(frame_def)
        hypotheses = []

        if subject and obj:
            hypotheses.append(f"{subject.text} and {obj.text} are involved in an event where {first_sentence}")
        if verb and obj:
            hypotheses.append(f"The {obj.text} undergoes an action where {first_sentence}")

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
        return hypotheses

    def get_nli_scores(self, model_key: str, premise: str, hypothesis: str) -> Dict[str, float]:
        model_info = self.models[model_key]
        inputs = model_info['tokenizer'](premise, hypothesis,
                                         return_tensors="pt",
                                         padding=True,
                                         truncation=True,
                                         max_length=512)

        with torch.no_grad():
            outputs = model_info['model'](**inputs)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)

        return {
            'entailment': scores[0][0].item(),
            'neutral': scores[0][1].item(),
            'contradiction': scores[0][2].item()
        }

    def analyze_csv(self, input_file: str, output_file: str):
        df = pd.read_csv(input_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            for idx, row in df.iterrows():
                sentence = row['Sentence']
                frame_def = row['Frame Definition']
                ground_truth = row['Ground Truth (Manually Checking)']

                f.write(f"\n{'= ' *80}\n")
                f.write(f"Analysis for Frame #{row['Frame #']} of Sentence ID {row['ID']}\n")
                f.write(f"{'= ' *80}\n")
                f.write(f"Original Sentence: {sentence}\n")
                f.write(f"Frame: {row['Envoked Frame']}\n")
                f.write(f"Frame Definition: {frame_def}\n")
                f.write(f"Ground Truth: {ground_truth}\n\n")

                # Basic Method Analysis
                basic_hyp = self.create_basic_hypothesis(frame_def)
                f.write("BASIC METHOD\n")
                f.write(f"Full NLI Input:\n")
                f.write(f"Premise: {sentence}\n")
                f.write(f"Hypothesis: {basic_hyp}\n\n")

                for model_key in ['microsoft', 'moritzlaurer']:
                    scores = self.get_nli_scores(model_key, sentence, basic_hyp)
                    f.write(f"{model_key.upper()} Model Scores:\n")
                    f.write(f"Entailment: {scores['entailment']:.4f}\n")
                    f.write(f"Neutral: {scores['neutral']:.4f}\n")
                    f.write(f"Contradiction: {scores['contradiction']:.4f}\n\n")

                # Advanced Method Analysis
                advanced_hyps = self.create_advanced_hypothesis(sentence, frame_def)
                f.write("\nADVANCED METHOD\n")

                for i, hyp in enumerate(advanced_hyps, 1):
                    f.write(f"\nVariant {i}:\n")
                    f.write(f"Full NLI Input:\n")
                    f.write(f"Premise: {sentence}\n")
                    f.write(f"Hypothesis: {hyp}\n\n")

                    for model_key in ['microsoft', 'moritzlaurer']:
                        scores = self.get_nli_scores(model_key, sentence, hyp)
                        f.write(f"{model_key.upper()} Model Scores:\n")
                        f.write(f"Entailment: {scores['entailment']:.4f}\n")
                        f.write(f"Neutral: {scores['neutral']:.4f}\n")
                        f.write(f"Contradiction: {scores['contradiction']:.4f}\n\n")

                f.write(f"{'= ' *80}\n\n")

def main():
    analyzer = NLIAnalyzer()
    analyzer.analyze_csv('nli_inspection.csv', 'nli_detailed_analysis.txt')

if __name__ == "__main__":
    main()




