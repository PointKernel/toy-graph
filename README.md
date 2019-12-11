# toy-graph
[![License:MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/l0icenses/MIT)

Toy code to test CUDA Graphs

## Build
```bash
make
```

## Run
mini-app to fake the real case workflow:
```bash
./testGraph [array-szie] (65536 by default)
```

graph & kernels runtime comparison:
```bash
./cmpRuntime
```

update graph parameters by directly setting parameters:
```bash
./setParamsNaive
```

update graph parameters by using APIs:
```bash
./setParamsAPI
```
