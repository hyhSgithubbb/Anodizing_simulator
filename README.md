# Anodizing_simulator
TiO2, Al2O3 아노다이징 공정을 시뮬레이션 해줄 수 있는 시뮬레이터 개발

전압·전류·전해질 레시피·음극재·공정 시간·전극 배치를 입력하면, 금속 표면의 위치별 산화막(Al₂O₃, TiO₂) 두께가 시간에 따라 어떻게 성장하는지 계산하고 실시간 그래프로 보여주는 GUI 프로그램입니다.

- 스택: Python, PyQt6, NumPy, PyQtGraph (+ SciPy 권장)
- 설계 문서: [docs/DESIGN.md](docs/DESIGN.md) (물리·화학 모델 수식 구조, 프로그램 아키텍처, 개발 로드맵)
