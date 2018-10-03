#PBS -S /bin/bash
#PBS -A ACF-UTHSC0007 
#PBS -l nodes=1,walltime=00:10:00 

module load cuda
module load julia

cd $PBS_O_WORKDIR
echo "printing pwd"
pwd
echo "printing PBS_O_WORKDIR"
echo $PBS_O_WORKDIR

WORKDIR=/lustre/haven/user/xiaoqihu/hg/fastlmm-gpu/test/

cd $WORKDIR

~/julia-0.7.0/bin/julia ./genome-scan-cpu.jl > genome-scan-output.txt

