# CRSA Architecture Scaling

Completed **46** validation-only atomic runs. Every comparison is paired by seed and scale value; the test split remained unopened.

## Width scale

|width|#|program|runs|validation bpb|paired Δ vs all-softmax|wins|parameters|tokens/s|
|---:|---:|---|---:|---:|---:|---:|---:|---:|
|32|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|2|4.918414 ± 0.105558|-0.036541 ± 0.001800|2/2|44672|40883 ± 8098|
|32|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.921675 ± 0.107673|-0.033281 ± 0.003916|2/2|44672|9607 ± 592|
|32|3|`softmax -> softmax`|2|4.954956 ± 0.103757|+0.000000 ± 0.000000|0/2|44672|26193 ± 14343|
|64|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|2|4.220917 ± 0.121794|-0.179606 ± 0.030003|2/2|138496|22319 ± 3032|
|64|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.224777 ± 0.118852|-0.175746 ± 0.027061|2/2|138496|12503 ± 367|
|64|3|`softmax -> softmax`|2|4.400523 ± 0.091791|+0.000000 ± 0.000000|0/2|138496|17363 ± 3349|
|96|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax`|2|4.007860 ± 0.120939|-0.233798 ± 0.044517|2/2|281472|11329 ± 1293|
|96|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.017937 ± 0.114749|-0.223721 ± 0.038327|2/2|281472|9054 ± 4416|
|96|3|`softmax -> softmax`|2|4.241658 ± 0.076422|+0.000000 ± 0.000000|0/2|281472|12739 ± 1156|

## Depth scale

|depth|#|program|runs|validation bpb|paired Δ vs all-softmax|wins|parameters|tokens/s|
|---:|---:|---|---:|---:|---:|---:|---:|---:|
|1|1|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|2|4.464803 ± 0.113203|-0.119118 ± 0.006082|2/2|57360|16860 ± 9503|
|1|2|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]`|2|4.465942 ± 0.110242|-0.117978 ± 0.003120|2/2|57360|31684 ± 22464|
|1|3|`softmax`|2|4.583920 ± 0.107122|+0.000000 ± 0.000000|0/2|57360|38047 ± 14030|
|3|1|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.463059 ± 0.108198|-0.139196 ± 0.000676|2/2|113520|5169 ± 195|
|3|2|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax -> softmax`|2|4.463736 ± 0.105447|-0.138518 ± 0.003427|2/2|113520|13566 ± 6910|
|3|3|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|2|4.464999 ± 0.108349|-0.137255 ± 0.000525|2/2|113520|4734 ± 1213|
|3|4|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax -> softmax`|2|4.467041 ± 0.105522|-0.135213 ± 0.003351|2/2|113520|7229 ± 1157|
|3|5|`softmax -> softmax -> softmax`|2|4.602254 ± 0.108874|+0.000000 ± 0.000000|0/2|113520|21456 ± 2250|
|4|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8] -> softmax -> softmax -> softmax`|2|4.482323 ± 0.137448|-0.131720 ± 0.025061|2/2|141600|11352 ± 4296|
|4|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax -> softmax`|2|4.484059 ± 0.134872|-0.129983 ± 0.022485|2/2|141600|7134 ± 3692|
|4|3|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax -> softmax -> softmax`|2|4.487916 ± 0.134044|-0.126126 ± 0.021657|2/2|141600|9487 ± 3106|
|4|4|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|2|4.494590 ± 0.117491|-0.119452 ± 0.005104|2/2|141600|8270 ± 2882|
|4|5|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8] -> softmax`|2|4.495004 ± 0.116846|-0.119038 ± 0.004459|2/2|141600|5374 ± 1000|
|4|6|`softmax -> softmax -> softmax -> softmax`|2|4.614043 ± 0.112387|+0.000000 ± 0.000000|0/2|141600|8002 ± 242|
