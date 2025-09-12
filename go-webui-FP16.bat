
set HF_HOME=%CD%\checkpoints

indextts2runtime\python.exe webui.py --host 127.0.0.1 --fp16 --deepspeed --cuda_kernel
pause