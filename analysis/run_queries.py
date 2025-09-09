import pandas as pd
import duckdb
from pathlib import Path
import yaml
import numpy as np
import csv
import math
from analysis import generate_latex, generate_plots

def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_hydra_df(df, prefix='cfg', sep='.'):
    rows = []
    for _, row in df.iterrows():
        job_id = row["slurm_job_id"]
        cfg_path = Path(row["output_directory"]) / ".hydra" / "config.yaml"
        
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        
        flat_cfg = flatten_dict(cfg, sep=sep)

        prefix_str = f"{prefix}{sep}" if prefix else ""
        flat_cfg_prefixed = {f"{prefix_str}{k}": v for k, v in flat_cfg.items()}
        flat_cfg_prefixed["slurm_job_id"] = job_id
        rows.append(flat_cfg_prefixed)

    return pd.DataFrame(rows).set_index("slurm_job_id")

def check_contains(value, target):
    if isinstance(value, (list, tuple, set)):
        return target in value
    elif pd.isna(value):
        return np.nan
    else:
        return False


if __name__ == "__main__":
    df_pt = pd.read_csv('/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/pretraining_runs.csv')
    df_tf = pd.read_csv('/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/transfer_runs.csv')
    df_tf['pretrain_id'] = df_tf['pretrain_id'].astype('Int64')
    df_tf['slurm_job_id'] = df_tf['slurm_job_id'].astype('Int64')
    df_pt['slurm_job_id'] = df_pt['slurm_job_id'].astype('Int64')

    df_pt_cfg = get_hydra_df(df_pt, prefix='cfg')
    df_tf_cfg = get_hydra_df(df_tf, prefix='cfg')

    df_pt_cfg = df_pt_cfg.copy()
    df_tf_cfg = df_tf_cfg.copy()

    df_pt_cfg['cfg.supervised'] = df_pt_cfg['cfg.loss.dense.use_labels'] == True
    df_pt_cfg['cfg.all_views'] = df_pt_cfg['cfg.loss.dense._target_'] == "pretraining.losses.supervised_loss.SupervisedContrastiveLoss"
    df_pt_cfg['cfg.pe_cf'] = df_pt_cfg['cfg.data.train.dataset.cf_suffixes'].apply(lambda x: check_contains(x, 'pe'))
    df_pt_cfg['cfg.sc_cf'] = df_pt_cfg['cfg.data.train.dataset.cf_suffixes'].apply(lambda x: check_contains(x, 'sc'))
    df_pt_cfg['cfg.aug.rotate'] = (df_pt_cfg['cfg.data.train.augmentation.rotate'] != False) & (df_pt_cfg['cfg.data.train.augmentation.rotate'] != 0)
    df_pt_cfg['cfg.aug.crop'] = (df_pt_cfg['cfg.data.train.augmentation.crop'] != False) & (df_pt_cfg['cfg.data.train.augmentation.rotate'] != 1)
    df_pt_cfg['cfg.aug.jitter'] = (df_pt_cfg['cfg.data.train.augmentation.jitter'] != False) & (df_pt_cfg['cfg.data.train.augmentation.jitter'] != 0)
    df_pt_cfg['cfg.aug.gaussian_blur'] = (df_pt_cfg['cfg.data.train.augmentation.gaussian_blur'] != False) & (df_pt_cfg['cfg.data.train.augmentation.gaussian_blur'] != 0)
    df_pt_cfg['cfg.aug.solarize'] = (df_pt_cfg['cfg.data.train.augmentation.solarize'] != False) & (df_pt_cfg['cfg.data.train.augmentation.solarize'] != 0)

    df_pt_with_cfg = pd.merge(
        df_pt,
        df_pt_cfg,
        left_on="slurm_job_id",
        right_on="slurm_job_id",
        how="inner"
    )

    df_tf_with_cfg = pd.merge(
        df_tf,
        df_tf_cfg,
        left_on="slurm_job_id",
        right_on="slurm_job_id",
        how="inner"
    )

    tf_prefixed = df_tf_with_cfg.add_prefix('transfer.')
    pt_prefixed = df_pt_with_cfg.add_prefix('pretraining.')
    tf_as_pretrain_prefixed = df_tf_with_cfg.add_prefix('pretraining.')

    def merge_with_pretrain(transfer_df, pretrain_df):
        return (
            pd.merge(
                transfer_df,
                pretrain_df,
                left_on="transfer.pretrain_id",
                right_on="pretraining.slurm_job_id",
                how="inner",
                validate="many_to_one"
            )
            .drop(columns=["pretraining.pretrain_id"], errors="ignore")
        )

    df1 = merge_with_pretrain(tf_prefixed, tf_as_pretrain_prefixed)
    df2 = merge_with_pretrain(tf_prefixed, pt_prefixed)

    df3 = tf_prefixed[tf_prefixed['transfer.pretrain_id'].isna()]

    df = pd.concat([df1, df2, df3], ignore_index=True)
    df = df[df['transfer.slurm_job_id'] != 10708]


    queries_dir = Path("/vol/biomedic3/bglocker/mscproj/mal224/DCCL/analysis/queries")
    output_dir = Path("/vol/biomedic3/bglocker/mscproj/mal224/DCCL/analysis/results")

    for query_file in queries_dir.glob("*.sql"):
        with open(query_file) as f:
            sql_query = f.read()

        result_df = duckdb.query(sql_query).df()

        out_name = query_file.stem + ".csv"
        result_df.to_csv(output_dir / out_name, index=False, quoting=csv.QUOTE_STRINGS)
    
    generate_latex.main()
    generate_plots.main()