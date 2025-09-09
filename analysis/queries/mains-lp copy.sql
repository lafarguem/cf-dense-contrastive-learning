SELECT
    "transfer.slurm_job_id" AS ID,
    "pretraining.best_probe_dice" AS "DSC mean",
    "pretraining.best_dist_to_var_ratio" AS "$d$/$\sigma$ mean",
    "pretraining.best_k_means_accuracy" AS "K-means purity mean"
FROM df
WHERE "transfer.slurm_job_id" IN (10646,10916,10729,10656)