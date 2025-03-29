import subprocess
import os
from datetime import datetime
import argparse
import subprocess
import json


def launch_sbatch(
    config,
    m,
    k,
    n,
    num_gpus,
    algorithm,
    total_sms,
    streamk_sms,
    comm_tile_m,
    comm_tile_n,
    hash,
    sbatch_script_content,
):

    job_name = f"{hash}/{algorithm}_{m}-{k}-{n}_{num_gpus}"

    slurm_out_dir = f"slurm_logs/{job_name}"
    if not os.path.exists(slurm_out_dir):
        os.makedirs(slurm_out_dir, exist_ok=True)
        os.makedirs(slurm_out_dir + "/" + hash, exist_ok=True)

    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    output_json = os.path.join(
            "../slurm_logs", job_name, f"{job_name}_{timestamp}.json"
        )
    output_log = os.path.join(
            "../slurm_logs", job_name, f"{job_name}_{timestamp}.log"
        )

    formatted_script = sbatch_script_content.format(
        job_name=job_name,
        image_name=config["image_name"],
        partition=config["partition"],
        m=m,
        n=n,
        k=k,
        num_gpus=num_gpus,
        algorithm=algorithm,
        time_limit=config["time_limit"],
        exclude_list=config["exclude_list"],
        total_sms=total_sms,
        streamk_sms=streamk_sms,
        hash=hash,
        COMMUNICATION_TILE_M=comm_tile_m,
        COMMUNICATION_TILE_N=comm_tile_n,
        output_json_file=output_json,
        output_log_file=output_log,
    )

    sbatch_script_path = os.path.join(
        "slurm_logs", job_name, f"{job_name}_{timestamp}.sbatch"
    )
    with open(sbatch_script_path, "w") as sbatch_file:
        sbatch_file.write(formatted_script)
    print(f"SBATCH script saved at: {sbatch_script_path}")
    print(f"Output JSON at: {output_json}")
    print(f"Output log at: {output_log}")

    try:
        if config["partition"] is None:
            os.chmod(sbatch_script_path, 0o755)
            subprocess.run(
                sbatch_script_path,
                capture_output=True,
                text=True,
                check=True,
            )
        else:
            subprocess.run(
                ["sbatch", sbatch_script_path],
                capture_output=True,
                text=True,
                check=True,
            )

        print(f"Successfully submitted job: {job_name}")
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job: {job_name}")
        print(f"Error message: {e.stderr}")


def main(hashes, config, sbatch_script_content):
    algorithms = ["all_reduce", "all_scatter", "one_shot"]

    dataset_file = "dataset/deepseek-coder-6.7b-base.json"
    with open(dataset_file, "r") as file:
        data = json.load(file)

    unique_mkn = list(set((entry["m"], entry["k"], entry["n"]) for entry in data))
    if config["partition"] != None:
        if "mi300" in config["partition"]:
            print("Running on MI300")
            total_sms = 304
            streamk_sms = 256
        elif "mi250" in config["partition"]:
            print("Running on MI250")
            total_sms = 104
            streamk_sms = 87
    else:
        print("Assuming MI300")
        total_sms = 304
        streamk_sms = 256

    enable_algorithms = False
    enable_streamk_sms = False
    enable_mkn = True
    enable_communication_tile = True

    algorithms_iter = algorithms if enable_algorithms else ["all_scatter"]
    streamk_sms_iter = [32, 64, 128, 256, 302, 304] if enable_streamk_sms else [streamk_sms]
    unique_mkn_iter = enumerate(unique_mkn) if enable_mkn else [(0, (4096, 11008, 65536))]

    communication_tile_m = [32, 64, 128, 256] if enable_communication_tile else [128]
    communication_tile_n = [32, 64, 128, 256] if enable_communication_tile else [128]


    for hash in hashes:
        for algorithm in algorithms_iter:
            for streamk_sms in streamk_sms_iter:
                for i, (m, k, n) in unique_mkn_iter:
                    for comm_tile_m in communication_tile_m:
                        for comm_tile_n in communication_tile_n:
                            max_gpus = 8
                            min_gpus = 1
                            num_gpus = min_gpus
                            print(f"Index: {i} / {len(unique_mkn)}, m: {m}, k: {k}, n: {n}")
                            while num_gpus <= max_gpus:
                                launch_sbatch(
                                    config,
                                    m,
                                    k,
                                    n,
                                    num_gpus,
                                    algorithm,
                                    total_sms,
                                    streamk_sms,
                                    comm_tile_m,
                                    comm_tile_n,
                                    hash,
                                    sbatch_script_content,
                                )
                                num_gpus *= 2


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process partition and commit hashes.")
    parser.add_argument(
        "--partition", nargs="?", default=None, help="The partition name (optional)"
    )
    parser.add_argument(
        "--commit_before", nargs="?", default=None, help="Commit hash before (optional)"
    )
    parser.add_argument(
        "--commit_after", nargs="?", default=None, help="Commit hash after (optional)"
    )
    args = parser.parse_args()
    partition = args.partition

    commit_before = args.commit_before or "latest"
    commit_after = args.commit_after or "latest"

    commit_hashes = list(dict.fromkeys([commit_before, commit_after]))

    config = {
        "image_name": "rocshmem_rocm_6.2.3.sif",
        "partition": partition,
        "time_limit": "00:05:00",
        "exclude_list": "",
    }

    sbatch_script_content = """#!/bin/bash
#SBATCH -J {job_name}                               # Job name
#SBATCH -o slurm_logs/{job_name}/{job_name}.%j.out  # Name of stdout output file (%j expands to jobId)
#SBATCH -N 1                                        # Total number of nodes requested
#SBATCH -n 128                                      # Total number of mpi tasks requested
#SBATCH -t {time_limit}                             # Run time (hh:mm:ss)
#SBATCH --partition={partition}                     # Partition
#SBATCH --exclude={exclude_list}                    # Exclude list (e.g., node[01,13-15])
image_path=./apptainer/images/{image_name}
num_gpus={num_gpus}
algorithm={algorithm}
m={m}
n={n}
k={k}
output_json_file={output_json_file}
output_log_file={output_log_file}
total_sms={total_sms}
streamk_sms={streamk_sms}
COMMUNICATION_TILE_M={COMMUNICATION_TILE_M}
COMMUNICATION_TILE_N={COMMUNICATION_TILE_N}
hash={hash}
echo "source /opt/conda/bin/activate py_3.10 &&\
    if [ \"${{hash}}\" != \"latest\" ]; then \
        git reset --hard ${{hash}}; \
    fi && \
    cd stream-k &&\
    timeout 5m mpirun --allow-run-as-root -np ${{num_gpus}}\
        python benchmark.py --algorithm ${{algorithm}}\
            -m ${{m}} -n ${{n}} -k ${{k}}\
                --total_sms ${{total_sms}}\
                --streamk_sms ${{streamk_sms}}\
                --validate --benchmark --debug\
                --heap_size 8589934592\
                --output_file ${{output_json_file}}\
                --COMMUNICATION_TILE_M ${{COMMUNICATION_TILE_M}}\
                --COMMUNICATION_TILE_N ${{COMMUNICATION_TILE_N}}\
        &> $output_log_file" \
    | apptainer exec --cleanenv ${{image_path}} bash
    """

    main(commit_hashes, config, sbatch_script_content)
