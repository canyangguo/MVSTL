## Comprehensive Spatiotemporal Learning for Traffic Forecasting

## make dir
```
mkdir params
mkdir log
```

## run
```
python main.py --config config/PEMS08.json --gpu 0 --model_name ours --st_dropout_rate 0.95 --noise_rate 2 --num_of_latents 32
```

## requirements
* python 3.9.12
* torch 1.11.0
* numpy 1.20.1
* pandas 1.4.2
