#PBS -S /bin/bash
#PBS -A ACF-UTHSC0007 
#PBS -l nodes=1,walltime=05:00:00:00,partition=skylake_volta 

module load cuda
module load julia
module load openBLAS

WORKDIR=/lustre/haven/user/xiaoqihu/hg/fastlmm-gpu/test/

cd $WORKDIR

julia0.7.0 ./dgemm.jl > gemm-timing/dgemm-acf-timing-result.txt

