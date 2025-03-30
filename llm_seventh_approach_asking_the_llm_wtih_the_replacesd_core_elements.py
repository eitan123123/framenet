import pandas as pd
import os
from typing import Dict, List, Tuple, Any, Optional, Set
import re
import nltk
from nltk.corpus import framenet as fn
import spacy
from openai import OpenAI
from sklearn.metrics import accuracy_score, recall_score, precision_score
from dotenv import load_dotenv
from tqdm import tqdm

# Download FrameNet data if not already available
try:
    fn.frames()
except LookupError:
    nltk.download('framenet_v17')

# Download spaCy model for POS tagging and dependency parsing
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class HybridFrameAnalyzer:
    def __init__(self, threshold=0.6, model="gpt-4-turbo-preview"):
        """Initialize the hybrid frame analyzer with OpenAI client"""
        # Initialize OpenAI client
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OpenAI API key not found in environment variables")
        self.openai_client = OpenAI(api_key=api_key)

        # Set LLM model and threshold
        self.llm_model = model
        self.ACCEPTANCE_THRESHOLD = threshold

    def get_first_sentence(self, text: str) -> str:
        """Extract the complete first sentence without truncation"""
        # Clean up parenthetical examples
        text = text.split("(e.g.")[0].split("(e")[0].strip()
        sentences = text.split('.')
        first_sent = sentences[0].strip().lower()
        first_sent = first_sent.split('(')[0].strip()
        if not first_sent.endswith('.'):
            first_sent += '.'
        return first_sent

    def get_core_elements_in_first_sentence(self, frame_name: str, frame_definition: str) -> List[Dict[str, Any]]:
        """Extract core frame elements that appear in the first sentence of the definition"""
        try:
            # First get all core elements from the frame
            all_core_elements = self.get_all_core_frame_elements(frame_name, frame_definition)

            # Extract first sentence
            first_sentence = self.get_first_sentence(frame_definition)

            # Filter to only include elements that appear in the first sentence
            elements_in_first_sentence = []
            for element in all_core_elements:
                # Check if element appears in the first sentence (with or without underscores)
                element_name = element['name']
                element_name_no_underscore = element_name.replace('_', ' ')

                if (element_name.lower() in first_sentence.lower() or
                        element_name_no_underscore.lower() in first_sentence.lower()):
                    elements_in_first_sentence.append(element)

            print(f"Found core elements in first sentence: {[elem['name'] for elem in elements_in_first_sentence]}")
            return elements_in_first_sentence

        except Exception as e:
            print(f"Error in get_core_elements_in_first_sentence: {str(e)}")
            return []

    def get_all_core_frame_elements(self, frame_name: str, frame_definition: str) -> List[Dict[str, Any]]:
        """Extract all core frame elements from a specified FrameNet frame"""
        try:
            # Clean up the frame name to match FrameNet conventions
            clean_frame_name = frame_name.replace(' ', '_').strip()

            # Get the frame by name
            frame = fn.frame_by_name(clean_frame_name)

            # Extract frame elements that have coreness set to "Core"
            core_elements = []
            for fe in frame.FE.values():
                if fe.coreType == "Core":
                    # Get the surrounding context from the frame definition
                    context = self.find_element_context(fe.name, frame_definition)

                    # Create a dictionary with the core element information
                    element_dict = {
                        'name': fe.name,  # Keep original name with underscores
                        'context': context
                    }

                    # Add additional information if available
                    if hasattr(fe, 'definition') and fe.definition:
                        element_dict['definition'] = fe.definition

                    if hasattr(fe, 'semType') and fe.semType:
                        element_dict['semantic_type'] = fe.semType.name

                    core_elements.append(element_dict)

            print(f"Found all core elements from FrameNet: {[elem['name'] for elem in core_elements]}")
            return core_elements

        except Exception as e:
            print(f"Error in get_all_core_frame_elements: {str(e)}")

            # Fallback to the regex-based extraction method if FrameNet lookup fails
            print("Falling back to regex-based extraction method")
            return self.extract_core_elements_regex(frame_definition)

    def find_element_context(self, element_name: str, frame_definition: str) -> str:
        """Find context for an element in the frame definition"""
        # Look for the element name in the definition (original format with underscores)
        if element_name in frame_definition:
            # Get a window of characters around the element mention
            idx = frame_definition.find(element_name)
            start_idx = max(0, idx - 50)
            end_idx = min(len(frame_definition), idx + len(element_name) + 50)
            return frame_definition[start_idx:end_idx].strip()
        else:
            # Try with spaces instead of underscores
            modified_name = element_name.replace('_', ' ')
            if modified_name in frame_definition:
                idx = frame_definition.find(modified_name)
                start_idx = max(0, idx - 50)
                end_idx = min(len(frame_definition), idx + len(modified_name) + 50)
                return frame_definition[start_idx:end_idx].strip()

        # If not found, return a portion of the definition
        first_sentence = frame_definition.split('.')[0] if '.' in frame_definition else frame_definition
        return first_sentence

    def extract_core_elements_regex(self, frame_definition: str) -> List[Dict[str, str]]:
        """Extract core elements from FrameNet definition using regex (fallback method)"""
        try:
            # Get original frame definition and first sentence
            first_sentence = self.get_first_sentence(frame_definition)
            original_sentence = frame_definition.split('.')[0].strip()

            # Find terms that are likely to be core elements (looking in original to preserve case)
            core_elements = []

            # Use regex to find capitalized terms that might be core elements
            # This pattern specifically looks for capitalized terms, including those with underscores
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
                            'name': element,  # Keep original format with underscores
                            'context': context
                        })

            print(f"Found core elements using regex: {[elem['name'] for elem in core_elements]}")
            return core_elements

        except Exception as e:
            print(f"Error in extract_core_elements_regex: {str(e)}")
            return []

    def map_elements_to_sentence(self, sentence: str, core_elements: List[Dict[str, Any]], frame_definition: str) -> \
            Dict[str, Optional[str]]:
        """Map FrameNet core elements to parts of the input sentence using LLM"""
        mappings = {}

        for element in core_elements:
            try:
                # Get context information
                context = element.get('context', '')
                definition = element.get('definition', '')

                # Add definition to context if available
                if definition and definition not in context:
                    context = f"{context}\nElement Definition: {definition}"

                prompt = (
                    f"As a FrameNet analyzer, examine this sentence and find what part (if any) "
                    f"corresponds to the given semantic role (core element).\n\n"
                    f"Input Sentence: {sentence}\n"
                    f"Core Element: {element['name']}\n"
                    f"Element Context: {context}\n\n"
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
                    model=self.llm_model,
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

    def correct_grammar_with_spacy(self, text: str) -> str:
        """Use spaCy to identify and correct subject-verb agreement issues"""
        try:
            # Parse the text with spaCy
            doc = nlp(text)

            # Extract tokens and their relationships
            tokens = [token for token in doc]

            # Build a new corrected sentence
            corrected_words = []
            skip_indices = set()

            for i, token in enumerate(tokens):
                # Skip tokens that have already been processed
                if i in skip_indices:
                    continue

                # Check for subject-verb relationship
                if token.dep_ == "nsubj" and i + 1 < len(tokens) and tokens[i + 1].pos_ == "VERB":
                    subject = token.text.lower()
                    verb = tokens[i + 1].text.lower()

                    # Apply subject-verb agreement rules
                    corrected_verb = self.get_correct_verb_form(subject, verb)

                    # Add the subject
                    corrected_words.append(token.text)

                    # Add the corrected verb
                    if corrected_verb != verb:
                        corrected_words.append(corrected_verb)
                    else:
                        corrected_words.append(tokens[i + 1].text)

                    # Mark the verb token as processed
                    skip_indices.add(i + 1)

                # For other tokens just add them as is
                elif i not in skip_indices:
                    corrected_words.append(token.text)

            # Handle special cases that might be missed by spaCy
            corrected_text = " ".join(corrected_words)
            corrected_text = self.handle_special_grammar_cases(corrected_text)

            return corrected_text

        except Exception as e:
            print(f"Error in correct_grammar_with_spacy: {str(e)}")
            # Fallback to basic grammar correction
            return self.handle_special_grammar_cases(text)

    def get_correct_verb_form(self, subject: str, verb: str) -> str:
        """Determine the correct verb form based on the subject"""
        # Irregular verb forms
        irregular_verbs = {
            'be': {'i': 'am', 'he': 'is', 'she': 'is', 'it': 'is', 'this': 'is', 'that': 'is',
                   'we': 'are', 'they': 'are', 'you': 'are', 'these': 'are', 'those': 'are'},
            'have': {'i': 'have', 'he': 'has', 'she': 'has', 'it': 'has', 'this': 'has', 'that': 'has',
                     'we': 'have', 'they': 'have', 'you': 'have', 'these': 'have', 'those': 'have'},
            'do': {'i': 'do', 'he': 'does', 'she': 'does', 'it': 'does', 'this': 'does', 'that': 'does',
                   'we': 'do', 'they': 'do', 'you': 'do', 'these': 'do', 'those': 'do'},
            'go': {'i': 'go', 'he': 'goes', 'she': 'goes', 'it': 'goes', 'this': 'goes', 'that': 'goes',
                   'we': 'go', 'they': 'go', 'you': 'go', 'these': 'go', 'those': 'go'},
        }

        # Handle irregular verbs
        if verb in irregular_verbs and subject in irregular_verbs[verb]:
            return irregular_verbs[verb][subject]

        # Handle regular verbs
        singular_subjects = ['he', 'she', 'it', 'this', 'that']
        plural_subjects = ['i', 'we', 'they', 'you', 'these', 'those']

        # For third person singular subjects
        if subject in singular_subjects:
            # If verb already ends with 's', no change needed
            if verb.endswith('s') and not verb.endswith('ss'):
                return verb
            # Add 's' to the verb for third person singular
            elif not verb.endswith('s'):
                return verb + 's'
            return verb

        # For plural subjects
        elif subject in plural_subjects:
            # Remove 's' from verbs for plural subjects (if not part of the root)
            if verb.endswith('s') and not verb.endswith('ss') and len(verb) > 2:
                return verb[:-1]
            return verb

        # Default case: return the original verb
        return verb

    def handle_special_grammar_cases(self, text: str) -> str:
        """Handle special grammar cases that might be missed"""
        # Replace common incorrect forms
        replacements = [
            (r'\bi goes\b', 'I go'),
            (r'\bi has\b', 'I have'),
            (r'\bi does\b', 'I do'),
            (r'\bI is\b', 'I am'),
            (r'\byou goes\b', 'you go'),
            (r'\byou has\b', 'you have'),
            (r'\byou does\b', 'you do'),
            (r'\byou is\b', 'you are'),
            (r'\bwe goes\b', 'we go'),
            (r'\bwe has\b', 'we have'),
            (r'\bwe does\b', 'we do'),
            (r'\bwe is\b', 'we are'),
            (r'\bthey goes\b', 'they go'),
            (r'\bthey has\b', 'they have'),
            (r'\bthey does\b', 'they do'),
            (r'\bthey is\b', 'they are'),
            (r'\bhe go\b', 'he goes'),
            (r'\bhe do\b', 'he does'),
            (r'\bhe have\b', 'he has'),
            (r'\bshe go\b', 'she goes'),
            (r'\bshe do\b', 'she does'),
            (r'\bshe have\b', 'she has'),
            (r'\bit go\b', 'it goes'),
            (r'\bit do\b', 'it does'),
            (r'\bit have\b', 'it has'),
        ]

        result = text
        for pattern, replacement in replacements:
            result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

        return result

    def create_nli_hypothesis(self, frame_first_sentence: str, element_mappings: Dict[str, Optional[str]]) -> str:
        """Create hypothesis by substituting mapped elements and correcting grammar"""
        hypothesis = frame_first_sentence

        # Make a list of replacements to perform
        replacements = []
        for element_name, mapped_text in element_mappings.items():
            if mapped_text:
                # Add original element name (with underscores)
                replacements.append((element_name, mapped_text))

                # Also add the version with spaces instead of underscores
                element_name_spaced = element_name.replace('_', ' ')
                if element_name_spaced != element_name:
                    replacements.append((element_name_spaced, mapped_text))

        # Sort replacements by length (descending) to avoid partial replacements
        replacements.sort(key=lambda x: len(x[0]), reverse=True)

        # Perform the replacements
        for element_name, mapped_text in replacements:
            # Use case-insensitive replacement with word boundaries
            pattern = r'\b' + re.escape(element_name) + r'\b'
            hypothesis = re.sub(pattern, mapped_text, hypothesis, flags=re.IGNORECASE)

        # Remove articles before "I" when used as a personal pronoun
        hypothesis = re.sub(r'\b(an|a)\s+I\b', 'I', hypothesis, flags=re.IGNORECASE)

        # Fix "an I or a I" pattern to just "an I"
        hypothesis = re.sub(r'\ban\s+I\s+or\s+a\s+I\b', 'an I', hypothesis, flags=re.IGNORECASE)

        # Fix duplicate words (words repeated with "or")
        hypothesis = re.sub(r'\b(\w+)\s+or\s+\1\b', r'\1', hypothesis, flags=re.IGNORECASE)

        # Apply grammar correction for subject-verb agreement
        hypothesis = self.correct_grammar_with_spacy(hypothesis)

        # Return the final hypothesis
        return hypothesis

    def check_frame_relevance_with_llm(self, sentence: str, frame_name: str, hypothesis: str,
                                       frame_definition: str) -> Tuple[bool, float]:
        """
        Determine if the frame is relevant to the sentence using LLM
        Returns a tuple of (is_relevant, confidence_score)
        """
        try:
            prompt = f"""Carefully analyze if this FrameNet frame matches the exact meaning and context of the given sentence.

Sentence to analyze: {sentence}

Frame Name: {frame_name}
Core Frame Definition with mapped elements: {hypothesis}
Full Frame Definition: {frame_definition}

Guidelines for matching:
1. The frame must match the EXACT meaning and context of the sentence
2. The verb usage in the sentence must align with the frame's core meaning
3. Consider the physical action and its direct effects only
4. Only match if the core semantic roles align perfectly
5. First, analyze how well the semantic roles in the frame map to elements in the sentence
6. Then, judge if the overall meaning of the frame applies to the sentence

First explain your reasoning about the match in 2-3 sentences, focusing on the semantic roles.
Then provide a confidence score from 0.0 to 1.0, where:
- Scores above 0.8 indicate high confidence that the frame matches
- Scores between 0.6-0.8 indicate moderate confidence
- Scores below 0.6 indicate the frame likely doesn't match

Format your answer as:
Reasoning: [Your reasoning]
Confidence: [score between 0.0-1.0]
"""

            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system",
                     "content": "You are a precise frame semantic analyzer that carefully evaluates semantic role mappings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            output = response.choices[0].message.content.strip()

            # Extract the confidence score
            score_match = re.search(r'Confidence:\s*(0\.\d+|1\.0)', output)
            if score_match:
                confidence = float(score_match.group(1))
            else:
                # Fallback if confidence not found
                if "high confidence" in output.lower():
                    confidence = 0.85
                elif "moderate confidence" in output.lower():
                    confidence = 0.7
                elif "low confidence" in output.lower():
                    confidence = 0.4
                else:
                    confidence = 0.5

            is_relevant = confidence >= self.ACCEPTANCE_THRESHOLD

            return is_relevant, confidence

        except Exception as e:
            print(f"Error in check_frame_relevance_with_llm: {str(e)}")
            return False, 0.0

    def create_specialized_output(self, results: List[Dict[str, Any]], output_csv: str):
        """Create specialized CSV output with only the specified columns"""
        try:
            # Define the columns for the specialized output
            specialized_data = []

            for result in results:
                specialized_data.append({
                    'Sentence': result['Sentence'],  # Original sentence
                    'Frame': result['Frame_Name'],  # Frame name
                    'Original_Frame_Sentence': result['First_Sentence'],  # First sentence without replacements
                    'Modified_Frame_Sentence': result['Hybrid_Hypothesis'],  # Modified sentence with replacements
                    'Confidence': result['Confidence_Score'],  # Confidence score
                    'Frame_Accepted': result['Frame_Accepted'],  # Whether frame was accepted
                    'LLM_Reasoning': result.get('LLM_Reasoning', '')  # LLM's reasoning
                })

            # Create and save the DataFrame
            specialized_df = pd.DataFrame(specialized_data)
            specialized_df.to_csv(output_csv, index=False)
            print(f"\nSpecialized results saved to {output_csv}")

        except Exception as e:
            print(f"Error in create_specialized_output: {str(e)}")

    def process_dataset(self, input_csv: str, output_csv: str):
        """Process the dataset and generate analysis results"""
        try:
            # Read dataset
            df = pd.read_csv(input_csv)

            # Initialize new columns
            new_columns = [
                'First_Sentence', 'Core_Elements', 'Element_Mappings',
                'Hybrid_Hypothesis', 'Confidence_Score', 'Frame_Accepted',
                'LLM_Reasoning', 'Ground_Truth'
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

                # Get core elements from frame definition that appear in the first sentence
                frame_name = row['Envoked Frame']
                core_elements = self.get_core_elements_in_first_sentence(frame_name, row['Frame Definition'])

                # Map elements to original sentence
                element_mappings = self.map_elements_to_sentence(
                    row['Sentence'],
                    core_elements,
                    first_sentence
                )

                # Create hypothesis with grammar correction by substituting mapped elements
                hypothesis = self.create_nli_hypothesis(first_sentence, element_mappings)

                # Check frame relevance with LLM
                is_relevant, confidence = self.check_frame_relevance_with_llm(
                    row['Sentence'],
                    frame_name,
                    hypothesis,
                    row['Frame Definition']
                )

                # Store results with proper dictionary formatting
                element_mappings_dict = {}
                for element in core_elements:
                    mapped_text = element_mappings.get(element['name'])
                    if mapped_text and mapped_text.lower() != 'none':
                        element_mappings_dict[element['name']] = mapped_text
                    else:
                        # Include elements mapped to None with explicit None value
                        element_mappings_dict[element['name']] = None

                results.append({
                    'Sentence': row['Sentence'],
                    'Verb': row['Verb'],
                    'Frame_Name': row['Envoked Frame'],
                    'Frame_Definition': row['Frame Definition'],
                    'First_Sentence': first_sentence,
                    'Core_Elements_Mappings': element_mappings_dict,
                    'Hybrid_Hypothesis': hypothesis,
                    'Confidence_Score': confidence,
                    'Frame_Accepted': is_relevant,
                    'Ground_Truth': row['Ground Truth (Manually Checking)'] == 'yes'
                })

                # Collect metrics data
                if row['Ground Truth (Manually Checking)']:
                    predictions.append('yes' if is_relevant else 'no')
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
                    'Hybrid_Hypothesis': '',
                    'Confidence_Score': '',
                    'Frame_Accepted': '',
                    'Ground_Truth': '',
                    'Notes': f'Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}'
                }])

                results_df = pd.concat([metrics_row, results_df], ignore_index=True)

            # Save standard results
            results_df.to_csv(output_csv, index=False)
            print(f"\nResults saved to {output_csv}")

            # Create specialized output
            specialized_output_csv = f"hybrid_specialized_results.csv"
            self.create_specialized_output(results, specialized_output_csv)

            return results

        except Exception as e:
            print(f"Error in process_dataset: {str(e)}")
            return []


def main():
    try:
        print("Initializing Hybrid Frame Analyzer...")
        load_dotenv()

        # Update this path to your input file
        input_file = "/Users/eitan/Desktop/framenet/datasets/gold_labeling.csv"
        output_csv = "results_hybrid_frame_analyzer.csv"

        # Create analyzer with default threshold of 0.6
        analyzer = HybridFrameAnalyzer(threshold=0.6, model="gpt-4-turbo-preview")
        analyzer.process_dataset(input_file, output_csv)

        print("\nAnalysis completed!")

    except Exception as e:
        print(f"\nError in main execution: {str(e)}")


if __name__ == "__main__":
    main()