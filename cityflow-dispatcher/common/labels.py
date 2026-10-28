# Correct / ideal label logic. A frame is correct for a model if its box recall against the
# reference (yolov8l) is >= RECALL_THRESHOLD. Ideal label = cheapest correct model.
import config
from common.boxes import box_recall


def frame_recalls(ref, dets_by_letter):
    """{letter: recall vs ref} for every model in dets_by_letter."""
    return {letter: box_recall(ref, dets, config.MATCH_IOU) for letter, dets in dets_by_letter.items()}


def ideal_letter(recalls):
    """Cheapest model (n < s < m < l) whose recall >= threshold. Falls back to l.
    Frames where l finds nothing have recall 1.0 for everyone, so n wins them."""
    for letter in config.MODEL_LETTERS:
        if letter in recalls and recalls[letter] >= config.RECALL_THRESHOLD:
            return letter
    return config.REFERENCE_LETTER
