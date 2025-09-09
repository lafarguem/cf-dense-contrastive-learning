SELECT
    "transfer.slurm_job_id" AS ID,
    "transfer.best_eval_hausdorff_95_nf" AS "Hausdorff 95 (nf) mean",
    "transfer.best_eval_hausdorff_95_nf_std" AS "Hausdorff 95 (nf) std",
    "transfer.best_eval_hausdorff_95_pe" AS "Hausdorff 95 (pe) mean",
    "transfer.best_eval_hausdorff_95_pe_std" AS "Hausdorff 95 (pe) std",
    "transfer.best_eval_asd_nf" AS "ASD (nf) mean",
    "transfer.best_eval_asd_nf_std" AS "ASD (nf) std",
    "transfer.best_eval_asd_pe" AS "ASD (pe) mean",
    "transfer.best_eval_asd_pe_std" AS "ASD (pe) std"
FROM df
WHERE "transfer.slurm_job_id" IN (10828,10751,10938,10646,10916,10729,10656,10674,10768)
  AND "transfer.best_eval_dice" IS NOT NULL
ORDER BY "transfer.best_eval_dice" DESC