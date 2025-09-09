SELECT
    "transfer.slurm_job_id" AS ID,
    "transfer.best_eval_dice" AS "DSC mean",
    "transfer.best_eval_dice_nf" AS "DSC (nf) mean",
    "transfer.best_eval_dice_pe" AS "DSC (pe) mean",
    "transfer.best_eval_dice_std" AS "DSC std",
    "transfer.best_eval_dice_nf_std" AS "DSC (nf) std",
    "transfer.best_eval_dice_pe_std" AS "DSC (pe) std",
    "transfer.best_eval_hausdorff_95" AS "Hausdorff 95 mean",
    "transfer.best_eval_hausdorff_95_std" AS "Hausdorff 95 std",
    "transfer.best_eval_asd" AS "ASD mean",
    "transfer.best_eval_asd_std" AS "ASD std"
FROM df
WHERE "transfer.slurm_job_id" IN (10828,10751,10938,10646,10916,10729,10656,10768,10674)
  AND "transfer.best_eval_dice" IS NOT NULL
ORDER BY "transfer.best_eval_dice" DESC