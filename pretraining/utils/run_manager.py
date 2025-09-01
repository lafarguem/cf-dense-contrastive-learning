import os
import fcntl
import pandas as pd
from pretraining.utils.distributed import is_main_process
import csv

_row = {}
_file = '/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/pretraining_runs.csv'

def set_file(file):
    if not is_main_process():
        return
    global _file
    _file = file

def setup_run(cfg, run):
    if not is_main_process():
        return
    job_id = os.environ.get('SLURM_JOB_ID', '')

    row = {
        'slurm_job_id': int(job_id),
        'output_directory': cfg.runtime.output_dir,
    }

    if (pretrain_id := cfg.runtime.get('pretrain_id')) is not None:
        row['pretrain_id'] = int(pretrain_id)
    if not cfg.train.debug.wandb_offline:
        row['wandb_url'] = run.url

    update_run(row)

def update_run(values: dict):
    if not is_main_process():
        return
    _row.update(values)

def consume_run():
    if not is_main_process():
        return
    global _row

    if os.path.exists(_file):
        with open(_file, 'r+') as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX)

                df = pd.read_csv(file)

                new_cols = [c for c in _row.keys() if c not in df.columns]
                for col in new_cols:
                    df[col] = ""

                for col in df.columns:
                    if col not in _row:
                        _row[col] = ""

                df = pd.concat([df, pd.DataFrame([_row], columns=df.columns)], ignore_index=True)

                file.seek(0)
                file.truncate()
                df.to_csv(file, index=False, quoting=csv.QUOTE_STRINGS)

            finally:
                fcntl.flock(file, fcntl.LOCK_UN)
    else:
        with open(_file, 'w') as file:
            try:
                fcntl.flock(file, fcntl.LOCK_EX)
                pd.DataFrame([_row]).to_csv(file, index=False, quoting=csv.QUOTE_STRINGS)
            finally:
                fcntl.flock(file, fcntl.LOCK_UN)

    _row = {}