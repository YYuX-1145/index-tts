
set HF_HOME=%CD%\checkpoints

indextts2runtime\python.exe webui.py --host 127.0.0.1 --use_deepspeed --cuda_kernel
pause