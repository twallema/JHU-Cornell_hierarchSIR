## Running on JHU Rockfish

### Installation 

1. Login to rockfish

```bash
ssh {YOUR ROCKFISH USERNAME}@login.rockfish.jhu.edu
```

2. Go to the working directory

```bash
cd scr4_struelo1/flepimop-code/twallema
```

3. Installation (do once)

    3a. Clone the Github repository (do once)

    ```bash
    git clone git@github.com:twallema/JHU-Cornell_hierarchSIR.git
    ```

    3b. Install the conda environment (do once)

    ```bash
    cd JHU-Cornell_hierarchSIR
    module load anaconda3
    conda env create -f hierarchSIR_env.yml
    ```

    3b-bis. Or alternatively update the conda environment (needed after adding a dependency),

    ```bash
    conda activate HIERARCHSIR
    conda env update -f hierarchSIR_env.yml --prune
    ```

    3c. Check if the JHU Rockfish cluster has the C++ `boost` libraries available,

    ```bash
    module avail boost
    ```

    Which it should, next, activate the boost module in your shell,

    ```bash
    module load boost
    ```

    Check where boost lives,

    ```bash
    echo $BOOST_ROOT
    ```

    Which should return a pointer to a path similar to the one below,

    ```bash
    /data/apps/extern/spack_on/gcc/9.3.0/boost/1.80.0-tqpiknljbrfjpc5p4axtn67oo74gitiu
    ```

    The environment variable `$BOOST_ROOT` is used to distinghuish between a local Linux machine and a cluster running Linux.

    3d. Install the `hierarchSIR` Python package **inside the conda environment** using,

    ```bash
    pip install -e . --force-reinstall
    ```

### Submitting jobs

1. Activate the anaconda environment

    ```bash
    module load anaconda3
    conda activate HIERARCHSIR
    ```

2. Activate the boost module

    ```bash
    module load boost
    ```

3. Set up Git

    3a. Checkout to the right branch

    ```bash
    cd JHU-Cornell_hierarchSIR
    git branch 
    git checkout <my_branch>
    ```

    3b. Make sure the branch is up-to-date.

    ```bash
    git checkout origin
    git pull
    ```

4. Submit the job to the cluster

    4a. Write a job submission script

    See `scripts/code/submit_job.sh` for an example.

    4b. Make sure the job script is executable

    ```bash
    chmod +x my_submision_script.sh
    ```

    4c. Submit the script to the `slurm` queue

    ```bash
    sbatch my_script.sh
    ```

    4d. Monitor your job

    ```bash
    squeue -u your_username
    scancel --name=your_job_name
    ```

### Cluster Tips and Tricks

1. Reset all changes made on cluster in git:

```bash
git reset --hard && git clean -f -d
```

2. Copy files from the HPC to local computer.

    - Open a terminal where you want to place the files on your computer.
    - Run

    ```bash
    scp -r <username>@rfdtn1.rockfish.jhu.edu:/home/<username>/.ssh/<key_name.pub> .
    ```
