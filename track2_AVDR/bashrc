# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
	. /etc/bashrc
fi

#export PATH="/work/asr/dyliu2/AutoML/anaconda3/bin:$PATH"
#source /work/asr/dyliu2/AutoML/anaconda3/etc/profile.d/conda.sh
export CUDA_PATH=/opt/lib/cuda-10.1
export PATH="/home/intern/glzhong/ICU/bin:/work/asr/dyliu2/tools:/home/intern/glzhong/bin:$PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib64:$CUDA_PATH/lib64:$CUDA_PATH/lib:/home/intern/glzhong/bin:/home/intern/glzhong/bin/lib:/ps/sppro/dyliu2/workspace/openasr/lib:/home/intern/glzhong/ICU/lib
export CUDA_HOME="$CUDA_PATH"

export MODULEPATH=/opt/tool/modulefiles
module unload gcc/4.7.2
module unload gcc/4.8.5
module unload gcc/5.5-os7
module unload cuda/7.0-v4
module unload cuda/7.5-v5-rc
module unload cuda/8.0
module unload cuda/8.0-v6.0
module unload cuda/8.0-v6.0.21
module unload cuda/8.0.61-v7.0
module unload cuda/9.0-v7.0
module unload python/2.7.2

module load cuda/10.0-cudnn-7.6.5
#module load gcc/gcc/4.8.2-os7.2
module load gcc/7.3.0-os7.2
#module load python/3.7.4

#source activate pytorch
alias rm=/bin/rmtrash.sh