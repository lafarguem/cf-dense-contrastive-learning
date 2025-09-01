import argparse
import json
import subprocess
import os

def find_index(lst, value):
    try:
        return lst.index(value)
    except ValueError:
        return None

def run_script(
        args=None,
        pretraining_experiment=None, 
        transfer_experiment=None, 
        pretraining_slurm_args=None,
        transfer_slurm_args=None,
        pretraining_ablation=None,
        transfer_ablation=None,
        exp_name=None,
        cwd = None,
    ):
    command = ["bash", "./DCCL/slurm/run_full_pipeline.sh"]

    pretraining_overrides_list = []
    if pretraining_experiment is not None:
        pretraining_overrides_list.append(f"+experiment={pretraining_experiment}")
    if pretraining_ablation is not None:
        for ablation in pretraining_ablation:
            pretraining_overrides_list.append(f"{ablation['key']}={ablation['value']}")
    if pretraining_overrides_list != []:
        pretraining_overrides_string = ' '.join(pretraining_overrides_list)
        command.append("--pretraining_overrides")
        command.append(pretraining_overrides_string)
    
    transfer_overrides_list = [] 
    if transfer_experiment is not None:
        transfer_overrides_list.append(f"+experiment={transfer_experiment}")
    else:
        transfer_overrides_list.append("+experiment=default")
    if transfer_ablation is not None:
        for ablation in pretraining_ablation:
            transfer_overrides_list.append(f"{ablation['key']}={ablation['value']}")
    if transfer_overrides_list != []:
        transfer_overrides_string = ' '.join(transfer_overrides_list)
        command.append("--transfer_overrides")
        command.append(transfer_overrides_string)

    if pretraining_slurm_args is not None:
        command.append("--pretraining_slurm_args")
        command.append(pretraining_slurm_args)

    if transfer_slurm_args is not None:
        command.append("--transfer_slurm_args")
        command.append(transfer_slurm_args)

    command.append("--exp_name")
    if exp_name is None:
        exp_name = pretraining_experiment
    if pretraining_ablation is not None:
        exp_name = f"{exp_name}_abl_{'_'.join([k['key'] for k in pretraining_ablation])}"
    elif transfer_ablation is not None:
        exp_name = f"{exp_name}_abl_{'_'.join([k['key'] for k in transfer_ablation])}"
    command.append(exp_name)

    subprocess.run(command, cwd=cwd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--root_dir", type=str, required=False,
                        default='/vol/biomedic3/bglocker/mscproj/mal224')
    parser.add_argument("--pretraining_nodelist", type=str, required=False, default=None)
    parser.add_argument("--pretraining_partition", type=str, required=False, default=None)
    parser.add_argument("--transfer_nodelist", type=str, required=False, default=None)
    parser.add_argument("--transfer_partition", type=str, required=False, default=None)

    args = parser.parse_args()
    cwd = args.root_dir

    with open(os.path.join(cwd, 'DCCL', 'full_experiments', f"{args.config}.json"), "r") as f:
        experiments = json.load(f)

    for pretraining in experiments['pretraining']:
        pretraining_experiment = pretraining.get('experiment')

        pretraining_slurm_args = pretraining.get('slurm_args')
        base_pretraining_slurm_args = [] if pretraining_slurm_args is None else pretraining_slurm_args.split(' ')
        if args.pretraining_partition is not None:
            idx = find_index(base_pretraining_slurm_args, '--partition')
            if idx is None:
                base_pretraining_slurm_args.append('--partition')
                base_pretraining_slurm_args.append(args.pretraining_partition)
            else:
                base_pretraining_slurm_args[idx+1] = args.pretraining_partition
        if args.pretraining_nodelist is not None:
            idx = find_index(base_pretraining_slurm_args, '--nodelist')
            if idx is None:
                base_pretraining_slurm_args.append('--nodelist')
                base_pretraining_slurm_args.append(args.pretraining_nodelist)
            else:
                base_pretraining_slurm_args[idx+1] = args.pretraining_nodelist
        pretraining_slurm_args = None if base_pretraining_slurm_args == [] else ' '.join(base_pretraining_slurm_args)

        exp_name = pretraining.get('exp_name')
        for transfer in experiments.get('transfer', [{}]):
            transfer_experiment = transfer.get('experiment')

            transfer_slurm_args = transfer.get('slurm_args')
            base_transfer_slurm_args = [] if transfer_slurm_args is None else transfer_slurm_args.split(' ')
            if args.transfer_partition is not None:
                idx = find_index(base_transfer_slurm_args, '--partition')
                if idx is None:
                    base_transfer_slurm_args.append('--partition')
                    base_transfer_slurm_args.append(args.transfer_partition)
                else:
                    base_transfer_slurm_args[idx+1] = args.transfer_partition
            if args.transfer_nodelist is not None:
                idx = find_index(base_transfer_slurm_args, '--nodelist')
                if idx is None:
                    base_transfer_slurm_args.append('--nodelist')
                    base_transfer_slurm_args.append(args.transfer_nodelist)
                else:
                    base_transfer_slurm_args[idx+1] = args.transfer_nodelist
            transfer_slurm_args = None if base_transfer_slurm_args is None else ' '.join(base_transfer_slurm_args)

            for pretraining_ablation in [None] + pretraining.get('ablations', []):
                run_script(args,
                    pretraining_experiment=pretraining_experiment,
                    transfer_experiment=transfer_experiment,
                    pretraining_slurm_args=pretraining_slurm_args,
                    transfer_slurm_args=transfer_slurm_args,
                    pretraining_ablation=pretraining_ablation,
                    transfer_ablation=None,
                    exp_name=exp_name
                )
            for transfer_ablation in transfer.get('ablations', []):
                run_script(args,
                    pretraining_experiment=pretraining_experiment,
                    transfer_experiment=transfer_experiment,
                    pretraining_slurm_args=pretraining_slurm_args,
                    transfer_slurm_args=transfer_slurm_args,
                    pretraining_ablation=None,
                    transfer_ablation=transfer_ablation,
                    exp_name=exp_name
                )