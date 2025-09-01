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
)
SELECT
    "pretraining.cfg.pe_cf" AS "Disease Counterfactuals",
    "pretraining.cfg.sc_cf" AS "Scanner Counterfactuals",
    "pretraining.cfg.aug.rotate" AS Rotation,
    "pretraining.cfg.aug.crop" AS Crop,
    "pretraining.cfg.aug.jitter" AS Effects,
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
FROM ranked
WHERE rn = 1 
  AND "pretraining.cfg.supervised" = TRUE
  AND "pretraining.cfg.all_views" = TRUE
  AND "transfer.best_eval_dice" IS NOT NULL
ORDER BY "transfer.best_eval_dice" DESC