#PBS -S /bin/bash
#PBS -A ACF-UTHSC0007 
#PBS -l nodes=1,walltime=00:10:00,partition=skylake_volta 

module load cuda
module load julia

WORKDIR=/lustre/haven/user/xiaoqihu/hg/fastlmm-gpu/test/

cd $WORKDIR

/nics/d/home/xiaoqihu/julia-0.7.0/bin/julia ./genome-scan-cpu.jl > genome-scan-output.txt

