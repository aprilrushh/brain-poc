# Brain — Modern Hopfield Memory + Qwen3.5-397B RAG

UmpaRumpa Brain Project — Modern Hopfield Network 기반 외부 연상 메모리 엔진을 LLM 추론에 통합.

## 핵심 알고리즘

Modern Hopfield retrieval:
    weights = softmax(beta * Q * K^T)   # K, Q 모두 normalized
    top_k_indices = topk(weights, k=5)

- beta = 50.0 (sharpness, 0.9809 의미집중도 검증값)
- Encoder: BGE-M3 (BAAI/bge-m3), FP16, normalize_embeddings=True
- LLM: Qwen3.5-397B-A17B (Apache 2.0, 397B total / 17B active MoE)

## 4-Phase 구조

Phase 0 사전 준비   : 맥북 + Together API           ~$10        파이프라인 검증
Phase 1 Closed Beta : H100x1 (Brain) + Together API ~$1,700/월  50명 베타
Phase 2 자체 호스팅  : B200x4 fp8                   ~$11,500    자체 운영 입증
Phase 3 외부 데모    : B200x4 또는 API              가변         외부 closed beta

상세: Notion "Brain 운영 4-Phase Plan v1"

## 디렉토리 구조

brain-inference/
  src/
    brain.py              - Modern Hopfield retrieval
    llm_client.py         - LLM_MODE swap (api / local)
    thinking_router.py    - Hybrid thinking (ON default + auto OFF)
    orchestrator.py       - query -> Brain -> LLM -> answer
  scripts/
    download_sherlock.py  - Project Gutenberg 4권 다운로드
    load_corpus.py        - TXT/PDF -> chunk -> BGE-M3 -> 적재
    test_e2e.py           - end-to-end 검증
  data/                   - 임베딩 + docs (gitignored)
  logs/                   - 실행 로그 (gitignored)
  lambda_backup/          - 이전 GH200 검증 코드 (reference)

## 설치

  .venv/bin/python -m pip install -r requirements.txt
  cp .env.example .env
  # .env 에 TOGETHER_API_KEY, OPENROUTER_API_KEY 입력

## 실행 (Phase 0)

  .venv/bin/python scripts/download_sherlock.py
  .venv/bin/python scripts/load_corpus.py
  .venv/bin/python scripts/test_e2e.py

## 환경변수 (.env)

  LLM_MODE       = api                            (api / local)
  LLM_API_BASE   = https://api.together.xyz/v1    API endpoint
  LLM_API_KEY    = (TOGETHER_API_KEY)             API 키
  LLM_MODEL      = Qwen/Qwen3.5-397B-A17B         모델명
  DATA_DIR       = ./data                         임베딩 저장 위치
  LOG_DIR        = ./logs                         로그 저장 위치

## License

Internal — UmpaRumpa Brain Project
