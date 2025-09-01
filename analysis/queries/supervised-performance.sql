WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                "pretraining.cfg.supervised",
                "pretraining.cfg.all_views",
                "pretraining.cfg.pe_cf",
                "pretraining.cfg.sc_cf",
                "pretraining.cfg.aug.rotate",
                "pretraining.cfg.aug.crop",
                "pretraining.cfg.aug.jitter",
                "pretraining.cfg.aug.gaussian_blur",
                "pretraining.cfg.aug.solarize"
            ORDER BY "transfer.best_eval_dice" DESC
        ) AS rn
    FROM df
),
parameter_rows AS (
    SELECT *,
        CASE
            WHEN "pretraining.cfg.pe_cf" = TRUE
                 AND "pretraining.cfg.sc_cf" = TRUE
                 AND "pretraining.cfg.aug.rotate" = TRUE
                 AND "pretraining.cfg.aug.crop" = TRUE
                 AND "pretraining.cfg.aug.jitter" = TRUE
                 AND "pretraining.cfg.supervised" = TRUE
                 AND "pretraining.cfg.all_views" = FALSE
            THEN 'S-DVD-CL'  -- All true
            WHEN "pretraining.cfg.pe_cf" = TRUE
                 AND "pretraining.cfg.sc_cf" = TRUE
                 AND "pretraining.cfg.aug.rotate" = TRUE
                 AND "pretraining.cfg.aug.crop" = TRUE
                 AND "pretraining.cfg.aug.jitter" = TRUE
                 AND "pretraining.cfg.supervised" = TRUE
                 AND "pretraining.cfg.all_views" = TRUE
            THEN 'S-MVD-CL'  -- All true
            WHEN "pretraining.cfg.pe_cf" = FALSE
                 AND "pretraining.cfg.sc_cf" = FALSE
                 AND "pretraining.cfg.aug.rotate" = FALSE
                 AND "pretraining.cfg.aug.crop" = FALSE
                 AND "pretraining.cfg.aug.jitter" = FALSE
                 AND "pretraining.cfg.supervised" = TRUE
                 AND "pretraining.cfg.all_views" = FALSE
            THEN 'S-DVD-CL nu'  -- All false
            WHEN "pretraining.cfg.pe_cf" = FALSE
                 AND "pretraining.cfg.sc_cf" = FALSE
                 AND "pretraining.cfg.aug.rotate" = FALSE
                 AND "pretraining.cfg.aug.crop" = FALSE
                 AND "pretraining.cfg.aug.jitter" = FALSE
                 AND "pretraining.cfg.supervised" = TRUE
                 AND "pretraining.cfg.all_views" = TRUE
            THEN 'S-MVD-CL nu'  -- All false
            WHEN "transfer.slurm_job_id" = 10751 THEN 'Segmentation'
            WHEN "transfer.slurm_job_id" = 10938 THEN 'SimCLR'
            WHEN "transfer.slurm_job_id" = 10828 THEN 'No augmentations'
        END AS config_label
    FROM ranked
    WHERE (rn = 1
      AND "transfer.best_eval_dice" IS NOT NULL)
      OR "transfer.slurm_job_id" IN (10751,10938,10828)
)
SELECT
    config_label AS "Pre-training",
    "transfer.best_eval_dice" AS "DICE mean",
    "transfer.best_eval_dice_pe" AS "DICE (pe) mean",
    "transfer.best_eval_dice_nf" AS "DICE (nf) mean",
    "transfer.best_eval_dice_std" AS "DICE std",
    "transfer.best_eval_dice_pe_std" AS "DICE (pe) std",
    "transfer.best_eval_dice_nf_std" AS "DICE (nf) std",
    "transfer.best_eval_hausdorff_95" AS "Hausdorff 95 mean",
    "transfer.best_eval_hausdorff_95_std" AS "Hausdorff 95 std",
    "transfer.best_eval_asd" AS "ASD mean",
    "transfer.best_eval_asd_std" AS "ASD std"
FROM parameter_rows
WHERE config_label IS NOT NULL
ORDER BY "transfer.best_eval_dice" DESC