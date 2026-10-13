"""Al / Ti 아노다이징 산화막 성장 시뮬레이터."""

import os as _os

# 엔진은 작은 행렬(N ≈ 100–400)을 스텝마다 수천 번 푼다. 이 크기에서 BLAS 다중 스레드는
# 이득 없이 CPU 를 모두 점유해(1 h L1 계산: 단일 스레드 CPU 1.6 s ↔ 4 스레드 CPU 7.6 s)
# GUI 와 함께 돌 때 계산을 오히려 느리게 한다. numpy 를 처음 import 하기 전에만 효과가 있으며,
# 사용자가 이미 지정한 값은 존중한다.
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
del _os, _var

__version__ = "0.3.0"
