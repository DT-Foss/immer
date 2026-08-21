# Marginal-Residual Variable-Lag Budget Curve

Completed **8** atomic runs. Training lags were sampled uniformly from **8..48**; evaluation used fixed lags **8, 16, 24, 32, 40, 48, 56, 64, 72, 80** at context 96.

## Aggregate

|#|foundation → readout|runs|train-range accuracy|unseen-lag accuracy|worst-lag accuracy|mean bits|tokens/s|
|---:|---|---:|---:|---:|---:|---:|---:|
|1|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.75] -> softmax`|2|0.9581|0.9363|0.8943|0.6262|9933 ± 1967|
|2|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.5] -> softmax`|2|0.9013|0.8856|0.7201|1.1808|10476 ± 2247|
|3|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.25] -> softmax`|2|0.5900|0.4871|0.4600|2.9621|11350 ± 2048|
|4|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.125] -> softmax`|2|0.5819|0.4859|0.4633|3.0003|9637 ± 1262|

## Accuracy by lag

|operator|8|16|24|32|40|48|56|64|72|80|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.75] -> softmax`|0.9827|0.9747|0.9700|0.9659|0.9610|0.8943|0.9503|0.9471|0.9360|0.9116|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.5] -> softmax`|0.9714|0.9603|0.9512|0.9273|0.8776|0.7201|0.8418|0.8999|0.9074|0.8933|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.25] -> softmax`|0.7353|0.6391|0.5888|0.5610|0.5555|0.4600|0.4999|0.4940|0.4840|0.4703|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.125] -> softmax`|0.7277|0.6244|0.5759|0.5537|0.5464|0.4633|0.5000|0.4923|0.4839|0.4672|

## Bits by lag

|operator|8|16|24|32|40|48|56|64|72|80|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.75] -> softmax`|0.2270|0.2941|0.3354|0.3810|0.4428|1.3403|0.6098|0.6653|0.8229|1.1437|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.5] -> softmax`|0.5130|0.5989|0.6653|0.8376|1.2351|2.9183|1.5501|1.0976|1.0985|1.2933|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.25] -> softmax`|1.7619|2.1509|2.4644|2.6756|2.7169|3.7712|3.2686|3.4319|3.5923|3.7873|
|`marginal_residual[dd=3,fh=1,s=0.8,ab=0.125] -> softmax`|1.8031|2.2218|2.5296|2.7196|2.7708|3.7265|3.2729|3.4731|3.6333|3.8520|
