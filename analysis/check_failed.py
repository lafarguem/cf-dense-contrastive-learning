import os
import pandas as pd

transfer_csv = "/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/transfer_runs.csv"
pretraining_csv = "/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/pretraining_runs.csv"
training_dir = "/vol/biomedic3/bglocker/mscproj/mal224/DCCL/outputs/full"

directory = os.fsencode(training_dir)

transfer_df = pd.read_csv(transfer_csv)
pretraining_df = pd.read_csv(pretraining_csv)

complete_pretraining = pretraining_df["output_directory"].apply(lambda x: x.split('/')[-2]).tolist()
complete_transfer = transfer_df["output_directory"].apply(lambda x: x.split('/')[-2]).tolist()

failed_pretrainings = []
failed_transfers = []

print("total runs", len(os.listdir(directory)))

for f in os.listdir(directory):
    dirname = os.fsdecode(f)

    if not dirname in complete_pretraining:
        failed_pretrainings.append(dirname)
    if not dirname in complete_transfer:
        failed_transfers.append(dirname)

print("failed pretrainings", len(failed_pretrainings))
for e in failed_pretrainings:
    if e in failed_transfers:
        print(e, "       transfer failed")
    else:
        print(e)
print("")

print("failed transfers", len(failed_transfers))
for e in failed_transfers:
    if e in failed_pretrainings:
        print(e, "         pretrained failed")
    else:
        print(e)