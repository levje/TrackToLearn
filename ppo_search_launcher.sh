# Description: This script is used to launch a hyperparameter search for the PPO algorithm.
be_safe=1   # Set to 1 to avoid launching jobs by mistake. Set at 0 at your own risk of
            # launching a lot of jobs.

lrs=(0.00001 0.00005 0.0005)
gammas=(0.5 0.75 0.90 0.95)
clips=(0.1)

# Count the number of process to launch
n_jobs=$(( ${#lrs[@]} * ${#gammas[@]} * ${#clips[@]} ))

# If the n_jobs is too high, exit the script.
if [[ $n_jobs -gt 20 ]]; then
    echo "The number of jobs to launch is too high (${n_jobs} > 20)."
    echo "Please reduce the number of hyperparameters to search."
    exit 1
fi

# Print the number of jobs to launch and prompt the user to continue. If the user does not
# wish to continue, stop the script.
echo "${n_jobs} jobs will be launched."
read -p "Continue? [y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
fi

# Iterate over each list of hyperparameters and for each possible combination, launch a new job
# by calling dummy_script.sh with the hyperparameters as arguments.
job_id=1
accept_all=0
for lr in "${lrs[@]}"; do
    for gamma in "${gammas[@]}"; do
        for clip in "${clips[@]}"; do
            # If be_safe is set to 1 and accept_all is set to 1,
            # ask the user to confirm the launch of the job.
            if [[ $be_safe -eq 1 && $accept_all -eq 0 ]]; then
                read -p "Launch job with lr=${lr}, gamma=${gamma}, clip=${clip}? [y/n/a/s] " -n 1 -r
                echo
                
                if [[ $REPLY =~ ^[Aa]$ ]]; then
                    echo "Launching all jobs."
                    accept_all=1
                elif [[ $REPLY =~ ^[Ss]$ ]]; then
                    echo "Skipping job with lr=${lr}, gamma=${gamma}, clip=${clip}."
                    continue
                elif [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    echo "Job not launched."
                    exit 1
                fi
            else
                echo "${job_id}/${n_jobs} => Queuing job with lr=${lr}, gamma=${gamma}, clip=${clip}."
            fi

            sbatch ismrm_ttl_ppo_for_search.sh $lr $gamma $clip
            job_id=$((job_id+1))
        done
    done
done