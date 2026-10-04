# Job templates

SLURM examples for the GPU steps. Account, partition and QoS are left to the site. Working directories come from
the environment variables read by tc3_vqa/paths.py:

    export TC3_VQA_WORK=/path/to/work            intermediate files
    export TC3_VQA_CORPUS=/path/to/corpus        raw/, processed/, index/
    export TC3_VQA_VIDEOS=/path/to/videos        <video_id>.mp4
    export TC3_VQA_FRAMES=/path/to/frames        frames named by frame id
    export TC3_VQA_RELEASE=/path/to/release      the public package
    export TC3_VQA_EXPERIMENTS=/path/to/experiments

Submit with --export=ALL so that the variables reach the job.
