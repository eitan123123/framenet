import nltk
from nltk.corpus import framenet as fn

# Download FrameNet data if not already available
try:
    fn.frames()
except LookupError:
    nltk.download('framenet_v17')


def get_core_frame_elements(frame_name):
    """
    Extract core frame elements from a specified FrameNet frame.

    Args:
        frame_name (str): The name of the frame to query (e.g., "Commerce_buy")

    Returns:
        list: A list of dictionaries containing information about core frame elements
    """
    try:
        # Get the frame by name
        frame = fn.frame_by_name(frame_name)

        # Extract frame elements that have coreness set to "Core"
        core_elements = []
        for fe in frame.FE.values():
            if fe.coreType == "Core":
                core_elements.append({
                    'name': fe.name,
                    'definition': fe.definition,
                    'semantic_type': fe.semType,
                })

        return core_elements

    except ValueError as e:
        print(f"Error: {e}")
        # Show available frames if the specified frame wasn't found
        print("\nAvailable frames include:")
        frames = fn.frames()
        for i, f in enumerate(frames[:10]):
            print(f"- {f.name}")
        print(f"... and {len(frames) - 10} more")
        return []


def show_frame_info(frame_name):
    """
    Display information about a frame including its definition and core elements.

    Args:
        frame_name (str): The name of the frame to display
    """
    try:
        frame = fn.frame_by_name(frame_name)
        print(f"Frame: {frame.name}")
        print(f"Definition: {frame.definition}")
        print("\nCore Frame Elements:")

        core_elements = get_core_frame_elements(frame_name)
        if core_elements:
            for i, element in enumerate(core_elements, 1):
                print(f"\n{i}. {element['name']}")
                print(f"   Definition: {element['definition']}")
                if element['semantic_type']:
                    print(f"   Semantic Type: {element['semantic_type'].name}")
        else:
            print("No core elements found for this frame.")

    except ValueError as e:
        print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    # Example: Get core elements for the "Commerce_buy" frame
    frame_name = input("Enter the name of the frame to query (e.g., Commerce_buy): ")
    show_frame_info(frame_name)

    # To get just the list of core elements programmatically:
    # core_elements = get_core_frame_elements(frame_name)