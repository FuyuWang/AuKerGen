cd src
CUDA_VISIBLE_DEVICES=0 python main.py --fitness1 latency --fitness2 latency --fitness3 energy --n_warmup_steps 2 --lr_mul 0.05 --batch_size 32 --epochs 15
cd ../
