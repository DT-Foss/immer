# Three-Seed Confirmation — Self–Local–Balanced–Free Attention

Configuration: context 96, width 48, 4 heads, 2 layers, 260 steps, batch 12. The file-hash test split is opened only in this confirmation stage.

|#|operator|seeds|validation bpb|test bpb|paired Δ vs softmax|seed wins|tokens/s|
|---:|---|---:|---:|---:|---:|---:|---:|
|1|`quad_route[dd=3,sh=0,lh=3,bh=0,s=0.8]`|3|4.527177 ± 0.021325|4.720256 ± 0.015395|-0.099746 ± 0.012807|3/3|34256 ± 12327|
|2|`quad_route[dd=3,sh=0,lh=2,bh=1,s=0.8]`|3|4.529020 ± 0.020105|4.726772 ± 0.010032|-0.093230 ± 0.007312|3/3|15497 ± 5572|
|3|`quad_route[dd=3,sh=0,lh=1,bh=1,s=0.8]`|3|4.536797 ± 0.016681|4.735188 ± 0.007419|-0.084814 ± 0.004023|3/3|17182 ± 7913|
|4|`softmax`|3|4.606658 ± 0.021474|4.820002 ± 0.003515|+0.000000 ± 0.000000|0/3|37084 ± 9562|
