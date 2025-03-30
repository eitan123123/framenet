import pandas as pd
from nltk.corpus import framenet as fn
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score
from openai import OpenAI
import os
from dotenv import load_dotenv


class ImprovedFrameMatcher:
    def __init__(self):
        """Initialize the frame matcher with OpenAI client"""
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.openai_client = OpenAI(api_key=api_key)

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
        """Extract the first sentence from frame definition"""
        # Split by example markers and take first part
        text = text.split("(e.g.")[0].split("(e")[0].strip()
        # Get first sentence
        sentences = text.split('.')
        first_sent = sentences[0].strip().lower()
        # Clean up any remaining parentheses
        first_sent = first_sent.split('(')[0].strip()
        # Add period if missing
        if not first_sent.endswith('.'):
            first_sent += '.'
        # Replace underscores with spaces for readability
        first_sent = first_sent.replace('_', ' ')
        return first_sent

    def check_frame_relevance(self, sentence: str, frame_name: str, frame_definition: str) -> bool:
        """Ask GPT to determine if the frame is relevant to the sentence with improved prompt"""
        try:
            # Extract first sentence of definition
            first_sentence = self.get_first_sentence(frame_definition)

            prompt = f"""Carefully analyze if this FrameNet frame matches the exact meaning and context of the given sentence.

Sentence to analyze: {sentence}

Frame Name: {frame_name}
Frame Core Definition: {first_sentence}
Full Frame Definition: {frame_definition}

Guidelines for matching:
1. The frame must match the EXACT meaning and context of the sentence
2. The verb usage in the sentence must align with the frame's core meaning
3. If there's any doubt, answer NO
4. Consider the physical action and its direct effects only
5. Don't match frames that only partially overlap with the sentence meaning

Answer YES only if there is a clear, direct match between the sentence's meaning and the frame's definition.
Answer NO if there is any uncertainty or if the match is only partial.

Your response must be exactly one word: either YES or NO.

Examples of correct matching:
1. Sentence: "John dropped his phone on the ground"
   Frame: "Cause_motion"
   Answer: YES (because dropping directly causes motion)

2. Sentence: "John dropped his phone on the ground"
   Frame: "Destroying"
   Answer: NO (because dropping doesn't necessarily result in destruction)

3. Sentence: "Mary broke the glass"
   Frame: "Cause_to_fragment"
   Answer: YES (breaking directly relates to fragmentation)

4. Sentence: "Mary broke the glass"
   Frame: "Cause_harm"
   Answer: NO (breaking an object isn't about causing harm to a person)

Now analyze this case:"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system",
                     "content": "You are a precise frame semantic analyzer. You must be very strict in matching frames to sentences, only answering YES when there is a clear, direct match."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )

            answer = response.choices[0].message.content.strip().upper()
            return answer == "YES"

        except Exception as e:
            print(f"Error in check_frame_relevance: {str(e)}")
            return False

    def process_dataset(self, input_csv: str, output_csv: str):
        """Process dataset using GPT frame relevance checking"""
        print(f"\nProcessing dataset: {input_csv}")
        df = pd.read_csv(input_csv)

        # Initialize new columns
        df['Frame Definition'] = ''
        df['GPT Frame Match'] = ''
        df['Original Frame'] = df['Envoked Frame']

        # Store results for each sentence
        results_dict = {}

        # Process unique sentences
        unique_sentences = df.drop_duplicates(subset=['ID', 'Sentence'])

        print("\nProcessing sentences...")
        total_sentences = len(unique_sentences)

        for idx, row in unique_sentences.iterrows():
            print(f"Processing sentence {idx + 1}/{total_sentences}: {row['Sentence']}")
            sentence_id = row['ID']
            sentence = row['Sentence']
            verb = row['Verb']

            frames = self.find_relevant_frames(verb)
            matches = []

            for frame_idx, frame in enumerate(frames, 1):
                print(f"  Checking frame {frame_idx}/{len(frames)}: {frame['name']}")

                is_relevant = self.check_frame_relevance(
                    sentence,
                    frame["name"],
                    frame["definition"]
                )

                if is_relevant:
                    matches.append({
                        "frame": frame["name"],
                        "definition": frame["definition"]
                    })

            results_dict[sentence_id] = matches

        # Update DataFrame with results
        for idx, row in df.iterrows():
            sentence_id = row['ID']
            frame_num = row['Frame #']
            current_frame = row['Envoked Frame']

            if sentence_id in results_dict:
                matches = results_dict[sentence_id]

                # Find if the current frame is in matches
                frame_match = any(match["frame"] == current_frame for match in matches)

                df.at[idx, 'GPT Frame Match'] = 'YES' if frame_match else 'NO'

                # Add frame definition if it's a match
                if frame_match:
                    frame_def = next(match["definition"] for match in matches
                                     if match["frame"] == current_frame)
                    df.at[idx, 'Frame Definition'] = frame_def

        # Calculate and print metrics
        df['Ground Truth Binary'] = df['Ground Truth (Manually Checking)'].str.lower().map({'yes': 1, 'no': 0})
        df['GPT Match Binary'] = df['GPT Frame Match'].map({'YES': 1, 'NO': 0})

        valid_predictions = df[df['Ground Truth (Manually Checking)'].str.lower().isin(['yes', 'no'])]

        if not valid_predictions.empty:
            # Calculate metrics
            accuracy = (valid_predictions['Ground Truth Binary'] ==
                        valid_predictions['GPT Match Binary']).mean()
            precision = precision_score(valid_predictions['Ground Truth Binary'],
                                        valid_predictions['GPT Match Binary'])
            recall = recall_score(valid_predictions['Ground Truth Binary'],
                                  valid_predictions['GPT Match Binary'])
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Print metrics
            print("\nModel Metrics:")
            print(f"Accuracy: {accuracy:.2%}")
            print(f"Precision: {precision:.2%}")
            print(f"Recall: {recall:.2%}")
            print(f"F1 Score: {f1:.2%}")

            # Print per-frame metrics
            print("\nPer-frame metrics:")
            for frame in valid_predictions['Envoked Frame'].unique():
                frame_data = valid_predictions[valid_predictions['Envoked Frame'] == frame]
                frame_accuracy = (frame_data['Ground Truth Binary'] ==
                                  frame_data['GPT Match Binary']).mean()
                print(f"{frame}: {frame_accuracy:.2%}")

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
                'GPT Frame Match': '',
                'Original Frame': '',
                'Ground Truth Binary': None,
                'GPT Match Binary': None
            }])

            df = pd.concat([metrics_row, df], ignore_index=True)

        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")


def run_analysis():
    """Run the improved frame analysis"""
    try:
        # Create output directory
        output_dir = "frame_analysis_results"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        input_file = "/Users/eitan/Desktop/framenet/datasets/gold_labeling.csv"
        output_file = os.path.join(output_dir, "results_gpt_improved.csv")

        # Run analysis
        matcher = ImprovedFrameMatcher()
        matcher.process_dataset(input_file, output_file)

        print("\nAnalysis completed!")
        print(f"Results saved in {output_file}")

    except Exception as e:
        print(f"\nError in analysis: {str(e)}")


if __name__ == "__main__":
    run_analysis()