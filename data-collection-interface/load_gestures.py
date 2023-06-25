# Written by Sem van den Broek on 03-05-2023
# Modified from gestures to digits by Gijs van de Linde on 10-05-2023
import os
import pickle
import re
from enum import Enum

DATA_DIR = "./dataset/digits"

class Hand(Enum):
    right = "right_hand"
    left = "left_hand"


class DigitNames(Enum):
    zero = "#0"
    one = "#1"
    two = "#2"
    three = "#3"
    four = "#4"
    five = "#5"
    six = "#6"
    seven = "#7"
    eight = "#8"
    nine = "#9"

class LoadGestureException(Exception):
    pass

def load_digits_for_candidate(candidate_name: str, digit_name: DigitNames, hand: Hand = Hand.right):
    result = []
    BASE_PATH = f"{DATA_DIR}/{digit_name.value}/{hand.value}"
    folder_items = os.listdir(BASE_PATH)

    # Filter on the .pickle extension
    filtered_ext = list(filter(lambda x: re.search(r'\.pickle$', x) is not None, folder_items))

    if len(filtered_ext) == 0:
        raise LoadGestureException("No gestures found in folder: %s" % BASE_PATH)

    for item in filtered_ext:
        r_match = re.match(r'candidate_(\w+).pickle$', item)
        if r_match is None:
            raise LoadGestureException("Incorrectly formatted data file name: %s" % item)
        candidate_data = []
        candidate_id = r_match.group(1)
        if candidate_name not in candidate_id:
            continue
        # Only open if candidate matches
        with open(os.path.join(BASE_PATH, item), 'rb') as f:
            while True:
                try:
                    data_contents = pickle.load(f)

                    if isinstance(data_contents, dict):
                        candidate_data.append(data_contents)
                    else:
                        # Old data loader
                        data = {
                            'data': data_contents,
                            'gesture': digit_name.value,
                            'candidate': candidate_id
                        }
                        candidate_data.append(data)
                except EOFError:
                    result.append((candidate_id, candidate_data))
                    break
    return result
