import nltk
from nltk.corpus import framenet as fn
from typing import List, Dict, Any
import spacy
import re

# Download FrameNet data if not already available
try:
    fn.frames()
except LookupError:
    nltk.download('framenet_v17')

# Load spaCy model for identifying verbs in sentences
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    import os

    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def get_frames_from_sentence(sentence: str) -> List[Dict[str, Any]]:
    """
    Find all potential frames in FrameNet for verbs in the given sentence.

    Parameters:
    sentence (str): The input sentence containing verbs

    Returns:
    List[Dict[str, Any]]: A list of potential frames for the sentence
    """
    # Extract verbs from the sentence using spaCy
    doc = nlp(sentence)
    verbs = [token.lemma_ for token in doc if token.pos_ == "VERB"]

    # List to store all frames
    all_frames = []

    # For each verb, get all matching frames
    for verb in verbs:
        # Clean the verb (remove whitespace, convert to lowercase)
        verb = verb.strip().lower()

        # Get all frames in FrameNet
        all_framenet_frames = fn.frames()

        # Go through each frame and check if the verb is a lexical unit
        for frame in all_framenet_frames:
            # Get all lexical units for this frame
            frame_name = frame.name
            lexical_units = frame.lexUnit.keys()

            # Check if any lexical unit matches our verb
            for lu_name in lexical_units:
                # The LU names are in format "word.pos", so we need to extract the word part
                if '.' in lu_name:
                    word = lu_name.split('.')[0].lower()
                    # Check if it matches our verb (exactly or as stem)
                    if word == verb or (len(word) > 3 and verb.startswith(word)):
                        # Get frame definition
                        frame_def = frame.definition if hasattr(frame, 'definition') else ""

                        # Get the first sentence of the definition
                        first_sentence = get_first_sentence(frame_def)

                        # Get core frame elements
                        core_elements = []
                        for fe in frame.FE.values():
                            if fe.coreType == "Core":
                                element_dict = {
                                    'name': fe.name,
                                    'definition': fe.definition if hasattr(fe, 'definition') else ""
                                }
                                core_elements.append(element_dict)

                        # Add to matching frames
                        frame_info = {
                            'name': frame_name,
                            'definition': frame_def,
                            'first_sentence': first_sentence,
                            'core_elements': core_elements,
                            'original_verb': verb,
                            'lexical_unit': lu_name
                        }

                        # Check if this frame is already in our list
                        if not any(f['name'] == frame_name for f in all_frames):
                            all_frames.append(frame_info)

                        # Break after finding a match in this frame
                        break

    return all_frames


def get_first_sentence(frame_definition: str) -> str:
    """Extract the complete first sentence from a frame definition"""
    if not frame_definition:
        return ""

    # Clean up parenthetical examples
    text = frame_definition.split("(e.g.")[0].split("(e")[0].strip()
    sentences = text.split('.')
    first_sent = sentences[0].strip().lower()
    first_sent = first_sent.split('(')[0].strip()
    if not first_sent.endswith('.'):
        first_sent += '.'
    return first_sent


def main():
    # Example usage
    sentence = "John broke the window"
    frames = get_frames_from_sentence(sentence)

    if not frames:
        print(f"No frames found for the verbs in: '{sentence}'")
    else:
        print(f"Found {len(frames)} potential frames for the sentence:")
        for i, frame in enumerate(frames, 1):
            print(f"\n{i}. Frame: {frame['name']}")
            print(f"   Triggered by verb: {frame['original_verb']}")

            # Print definition (truncated if too long)
            definition = frame['definition']
            if len(definition) > 100:
                print(f"   Definition: {definition[:100]}...")
            else:
                print(f"   Definition: {definition}")

            print(f"   First sentence: {frame['first_sentence']}")
            print(f"   Lexical unit: {frame['lexical_unit']}")
            print(f"   Core elements ({len(frame['core_elements'])}):")

            for elem in frame['core_elements']:
                elem_def = elem['definition']
                if elem_def and len(elem_def) > 80:
                    print(f"     - {elem['name']}: {elem_def[:80]}...")
                else:
                    print(f"     - {elem['name']}: {elem_def}")


if __name__ == "__main__":
    main()