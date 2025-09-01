SELECT
    "transfer.slurm_job_id" AS ID,
    "transfer.best_eval_dice" AS "DICE mean",
    "transfer.best_eval_dice_pe" AS "DICE (pe) mean",
    "transfer.best_eval_dice_nf" AS "DICE (nf) mean",
    "transfer.best_eval_dice_std" AS "DICE std",
    "transfer.best_eval_dice_pe_std" AS "DICE (pe) std",
    "transfer.best_eval_dice_nf_std" AS "DICE (nf) std"
FROM df
WHERE "transfer.slurm_job_id" IN (10828,10751,10938,10646,10916,10729,10656)
  AND "transfer.best_eval_dice" IS NOT NULL
ORDER BY "transfer.best_eval_dice" DESC