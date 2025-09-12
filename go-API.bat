
set HF_HOME=%CD%\checkpoints

indextts2runtime\python.exe indextts2_api.py --use_deepspeed --cuda_kernel
pause