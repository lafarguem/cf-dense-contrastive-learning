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
                "pretraining.cfg.aug.solarize",
                "pretraining.output_directory",
                "pretraining.cfg.train.epochs"
            ORDER BY "transfer.best_eval_dice" DESC
        ) AS rn
    FROM df
)
SELECT
    "pretraining.cfg.train.epochs" AS Epochs,
    "pretraining.cfg.supervised" AS Supervised,
    "pretraining.cfg.all_views" AS "All Views",
    "pretraining.cfg.pe_cf" AS "Disease Counterfactuals",
    "pretraining.cfg.sc_cf" AS "Scanner Counterfactuals",
    "pretraining.cfg.aug.rotate" AS Rotation,
    "pretraining.cfg.aug.crop" AS Crop,
    "pretraining.cfg.aug.jitter" AS Effects,
    "transfer.best_eval_dice" AS DSC,
    "transfer.best_eval_dice_pe" AS "DSC (pe)",
    "transfer.best_eval_dice_nf" AS "DSC (nf)",
    "pretraining.output_directory" AS "Output Directory (pretraining)",
    "transfer.output_directory" AS "Output Directory (transfer)",
    "pretraining.wandb_url" AS "WandB URL (pretraining)",
    "transfer.wandb_url" AS "WandB URL (transfer)"
FROM ranked
WHERE rn = 1
  AND "transfer.best_eval_dice" IS NOT NULL
  AND ("pretraining.cfg.supervised" = FALSE
    OR "pretraining.cfg.supervised" IS NULL)
ORDER BY "transfer.best_eval_dice_pe" DESC