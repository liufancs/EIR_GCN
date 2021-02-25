# EIR_GCN

## Environment Settings
- Tensorflow-gpu version:  1.3.0

## Example to run the codes.
# Toys&Games
Run EIR_GCN.py
```
python EIR_GCN.py --dataset Toys_and_Games  --regs [1e-3] --embed_size 64 --layer_size [64] --lr 0.0001 --batch_size 1024 --epoch 1000 --alpha 1.4 --Ks [20,10] --gpu_id 0
```

# Kinle Store
Run EIR_GCN.py
```
python EIR_GCN.py --dataset KS  --regs [1e-4] --embed_size 64 --layer_size [64] --lr 0.0005 --batch_size 1024 --epoch 1000 --alpha 1.2 --Ks [20,10] --gpu_id 0
```

# Moives
Run EIR_GCN.py
```
python EIR_GCN.py --dataset Movies  --regs [1e-3] --embed_size 64 --layer_size [64] --lr 0.0001 --batch_size 1024 --epoch 1000 --alpha 1.3 --Ks [20,10] --gpu_id 0
```

# Home&Kitchen
Run EIR_GCN.py
```
python EIR_GCN.py --dataset Home_and_Kitchen  --regs [1e-2] --embed_size 64 --layer_size [64] --lr 0.0001 --batch_size 1024 --epoch 1000 --alpha 1.0 --Ks [20,10] --gpu_id 0
```
