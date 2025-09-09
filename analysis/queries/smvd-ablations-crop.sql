WITH ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (
            PARTITION BY
                "pretraining.cfg.train.epochs",
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
            THEN 'All'  -- All true
            WHEN "pretraining.cfg.pe_cf" = TRUE
                 AND "pretraining.cfg.sc_cf" = TRUE
                 AND "pretraining.cfg.aug.rotate" = TRUE
                 AND "pretraining.cfg.aug.crop" = FALSE
                 AND "pretraining.cfg.aug.jitter" = TRUE
            THEN 'W/o crop'  -- All true except Disease CF
            WHEN "pretraining.cfg.pe_cf" = FALSE
                 AND "pretraining.cfg.sc_cf" = FALSE
                 AND "pretraining.cfg.aug.rotate" = FALSE
                 AND "pretraining.cfg.aug.crop" = FALSE
                 AND "pretraining.cfg.aug.jitter" = FALSE
            THEN 'No augmentations'  -- All false
            WHEN "pretraining.cfg.pe_cf" = FALSE
                 AND "pretraining.cfg.sc_cf" = FALSE
                 AND "pretraining.cfg.aug.rotate" = FALSE
                 AND "pretraining.cfg.aug.crop" = TRUE
                 AND "pretraining.cfg.aug.jitter" = FALSE
            THEN 'Crop only'  -- All false except Disease CF
        END AS config_label
    FROM ranked
    WHERE rn = 1
      AND "pretraining.cfg.supervised" = TRUE
      AND "pretraining.cfg.all_views" = TRUE
      AND "pretraining.cfg.train.epochs" = 20.0
      AND "transfer.best_eval_dice" IS NOT NULL
)
SELECT
    config_label AS "Augmentations",
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
        WHEN config_label = 'All' THEN 4
        WHEN config_label = 'W/o crop' THEN 3
        WHEN config_label = 'No augmentations' THEN 1
        WHEN config_label = 'Crop only' THEN 2
    END;
