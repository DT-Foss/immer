# Anchor-Residual Sinkhorn Foundation — Generation 12

Completed **15** atomic runs. Every candidate keeps a pure-softmax top layer and a whole free head inside the adaptive foundation layer.

## Archive byte corpus

Test split remained unopened.

|#|foundation → readout|runs|validation bpb|Δ vs F→F|effective S/L/B/F|tokens/s|
|---:|---|---:|---:|---:|---:|---:|
|1|`quad_route[sh=0,lh=3,bh=0,s=0.8] -> softmax`|3|3.799582 ± 0.052837|-0.250706|—|27505 ± 2546|
|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|3|3.811464 ± 0.063639|-0.238824|—|22299 ± 4020|
|3|`anchor_residual[dd=3,fh=1,s=0.8,ap=lll,ab=0.5] -> softmax`|3|3.813661 ± 0.052076|-0.236626|0.103/0.550/0.097/0.250|12837 ± 2029|
|4|`anchor_residual[dd=3,fh=1,s=0.8,ap=llb,ab=0.125] -> softmax`|3|3.814822 ± 0.061870|-0.235466|0.027/0.479/0.244/0.250|12650 ± 3164|
|5|`softmax -> softmax`|3|4.050288 ± 0.089370|+0.000000|—|32602 ± 3565|
