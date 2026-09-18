# Working directories, read from environment variables so that a run is configured without editing code.
import os

WORK = os.environ.get('TCCC_VQA_WORK', 'work')                     # intermediate artifacts of the construction pipeline
CORPUS = os.environ.get('TCCC_VQA_CORPUS', 'corpus')               # doctrine corpus: raw/, processed/, index/
VIDEOS = os.environ.get('TCCC_VQA_VIDEOS', 'videos')               # source videos as <video_id>.mp4
FRAMES = os.environ.get('TCCC_VQA_FRAMES', 'frames')               # every frame referenced by an item, named by frame id
RELEASE = os.environ.get('TCCC_VQA_RELEASE', 'release')            # the public dataset package
RELEASE_PARENT = os.path.dirname(os.path.abspath(RELEASE))
CANDIDATES = os.environ.get('TCCC_VQA_CANDIDATES', 'work/candidates')   # the item set before the recognition consensus
EXPERIMENTS = os.environ.get('TCCC_VQA_EXPERIMENTS', 'experiments')     # consensus votes, audits, and comparison runs
FIGURES = os.environ.get('TCCC_VQA_FIGURES', 'figures')
