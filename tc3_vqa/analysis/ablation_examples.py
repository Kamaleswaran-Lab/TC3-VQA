# The three public-domain examples of Figure 6: item, frame, crop, the reference pair, the pair written directly from the
# same frames, and the note under them.
from tc3_vqa.paths import RELEASE
from pathlib import Path

DEP = Path(RELEASE)

EXAMPLES = [
    dict(item='ans_00898', frame=0, crop=(0.0, 1.0), title='(a) Wrong intervention, plausible answer',
         ours=[('Recognition', 'Wound packing'),
               ('Q', 'How can the effectiveness of wound packing be improved according to TCCC guidance?'),
               ('A', '"The effectiveness of wound packing may be improved when skin closure over the packing can be achieved..."')],
         conv=[('Recognition', 'Hemostatic Dressing Application'),
               ('Q', 'What is the proper method for applying a tourniquet to control bleeding as shown here?'),
               ('A', 'Place tourniquet 2-3 inches above wound. Tighten until bleeding stops. Mark time...')],
         note='Wound packing is shown, not a tourniquet.'),
    dict(item='ans_00958', frame=1, crop=(0.0, 0.70), title='(b) Wrong intervention, unsupported answer',
         ours=[('Recognition', 'Surgical cricothyroidotomy'),
               ('Q', 'At what site should the initial vertical skin incision for a surgical cricothyroidotomy be made?'),
               ('A', '"Make vertical incision through the skin over cricothyroid membrane."')],
         conv=[('Recognition', 'Needle Decompression'),
               ('Q', 'What should be done if the needle decompression does not relieve tension pneumothorax?'),
               ('A', 'Perform chest tube insertion immediately.')],
         note='No needle decompression is shown.'),
    dict(item='ans_01089', frame=0, crop=(0.0, 1.0), title='(c) Wrong intervention, off-target care',
         ours=[('Recognition', 'Chest seal application'),
               ('Q', 'After applying a chest seal, what complication must the casualty be monitored for and how is it treated?'),
               ('A', '"Monitor the casualty for the potential development of a subsequent tension pneumothorax..."')],
         conv=[('Recognition', 'Chest Seal Application'),
               ('Q', "Given the visible chest wound and the application of a tourniquet, what is the next step in TCCC for managing this casualty's condition effectively?"),
               ('A', 'The next step is to check for airway obstruction and breathing issues... Ensure the tourniquet is properly secured...')],
         note='A chest seal is shown and no tourniquet is present.'),
]


def trim_black(im, thresh=18):
    """drop letterbox or pillarbox borders (rows or columns that are almost black) left by the source video"""
    import numpy as np
    a = np.asarray(im.convert('L'))
    cols = np.where(a.max(axis=0) > thresh)[0]; rows = np.where(a.max(axis=1) > thresh)[0]
    if len(cols) == 0 or len(rows) == 0:
        return im
    return im.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))
