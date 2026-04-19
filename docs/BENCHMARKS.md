# Benchmarks

Populate after running Phase 4 of the build plan. Template below.

## Methodology

- Dataset: 1,200 mixed photos (500 KB – 8 MB), formats {JPEG, PNG, HEIC}
- Command: `imgsearch index <fixture> --model siglip2-base --batch 16`
- Cold run: `imgsearch clean <fixture>` before each measurement
- Memory: `/usr/bin/time -l` peak RSS
- Latency: `time imgsearch search <fixture> -t "a red car"`

## Results

| Hardware | Cold index | Warm re-scan | Peak RSS | Search latency |
|---|---|---|---|---|
| M1 (8 GB) — batch 8 | TBD | TBD | TBD | TBD |
| M1 Pro (16 GB) | TBD | TBD | TBD | TBD |
| M2 (16 GB) | TBD | TBD | TBD | TBD |
| M3 Pro (18 GB) | TBD | TBD | TBD | TBD |

## Pass criteria

- 1,000 images indexed in < 120 s on M1 16 GB
- Peak RSS < 3 GB during indexing
- Search latency < 100 ms (including model warm-up: excluded)

## Known bottlenecks

- PIL decode dominates on large JPEGs. Consider threaded prefetch if warranted.
- First run downloads ~400 MB of model weights; not counted in "cold index".
