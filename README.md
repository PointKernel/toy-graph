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

update graph parameters by using APIs:
```bash
./setParamsAPI
```

update graph parameters by using pointer-of-pointer:
```bash
./setParamsWrapper
```
