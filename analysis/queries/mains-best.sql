WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                "pretraining.cfg.train.epochs",
                "pretraining.cfg.supervised",
                "pretraining.cfg.all_views"
            ORDER BY "transfer.best_eval_dice" DESC
        ) AS rn
    FROM df
),
parameter_rows AS (
    SELECT *,
        CASE
            WHEN "transfer.slurm_job_id" = 10751 THEN 'Segmentation'
            WHEN "transfer.slurm_job_id" = 10938 THEN 'SimCLR'
            WHEN "transfer.slurm_job_id" = 10828 THEN 'No augmentations'
            WHEN "transfer.slurm_job_id" = 10674 THEN 'SSDCL'
            WHEN "transfer.slurm_job_id" = 10768 THEN 'VADeR'
            WHEN "pretraining.cfg.supervised" = TRUE
                 AND "pretraining.cfg.all_views" = TRUE
            THEN 'S-MVD-CL'  -- All true
            WHEN "pretraining.cfg.supervised" = TRUE
                 AND "pretraining.cfg.all_views" = FALSE
            THEN 'S-DVD-CL'  -- All true
            WHEN "pretraining.cfg.supervised" = FALSE
                 AND "pretraining.cfg.all_views" = TRUE
            THEN 'MVD-CL'  -- All true
            WHEN "pretraining.cfg.supervised" = FALSE
                 AND "pretraining.cfg.all_views" = FALSE
            THEN 'DVD-CL'  -- All true
        END AS config_label
    FROM ranked
    WHERE (rn = 1
      AND "pretraining.cfg.train.epochs" = 20.0
      AND "transfer.best_eval_dice" IS NOT NULL)
      OR "transfer.slurm_job_id" IN (10751,10938,10828,10674,10768)
)
SELECT
    config_label AS "Pre-training",
    "transfer.best_eval_dice" AS "DSC mean",
    "transfer.best_eval_dice_pe" AS "DSC (pe) mean",
    "transfer.best_eval_dice_nf" AS "DSC (nf) mean",
    "transfer.best_eval_dice_std" AS "DSC std",
    "transfer.best_eval_dice_pe_std" AS "DSC (pe) std",
    "transfer.best_eval_dice_nf_std" AS "DSC (nf) std",
    "transfer.best_eval_hausdorff_95" AS "Hausdorff 95 mean",
    "transfer.best_eval_hausdorff_95_std" AS "Hausdorff 95 std",
    "transfer.best_eval_asd" AS "ASD mean",
    "transfer.best_eval_asd_std" AS "ASD std"
FROM parameter_rows
WHERE config_label IS NOT NULL
ORDER BY
    CASE
        WHEN config_label = 'S-DVD-CL' THEN 6
        WHEN config_label = 'S-MVD-CL' THEN 7
        WHEN config_label = 'DVD-CL' THEN 4
        WHEN config_label = 'MVD_CL' THEN 5
        WHEN config_label = 'No augmentations' THEN 1
        WHEN config_label = 'SimCLR' THEN 2
        WHEN config_label = 'Segmentations' THEN 3

    END;
