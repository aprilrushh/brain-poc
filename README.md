# Brain — Modern Hopfield Memory Engine for LLM Inference

UmpaRumpa **Brain** 프로젝트. Modern Hopfield Network 기반 외부 연상 메모리 엔진을 LLM 추론에 통합한 RAG 서비스. 유저가 PDF/TXT 파일을 업로드 → 질문 → 파일 내용에 근거한 답변 + 출처(citation)를 받는다.

**Production**: https://brain.umparumpa.com (라이브)
**Branch**: `d4-ui` (실서비스 추적 브랜치 — `main`은 무관한 옛 코드)
**Status**: Phase 1 closed beta launch-ready + Second Brain dev 병행

> ⚠️ 내부 개발 문서. private repo. 검증 수치·아키텍처는 경쟁 IP이므로 외부 공유 시 주의.

---

## 1. 핵심 기술

**Modern Hopfield retrieval** (Ramsauer et al. 2020, "Hopfield Networks is All You Need"):

```
weights = softmax(beta * Q @ K^T)   # Q, K 모두 normalized
top_w, top_i = topk(weights, k=top_k)
```

| 파라미터 | 값 | 비고 |
|---|---|---|
| beta (β) | 50.0 | sharpness, 의미집중도 0.9809 검증값 |
| top_k | 10 | 운영값 (v6 SYSTEM_PROMPT) |
| Encoder | BAAI/bge-m3 | FP16, normalize_embeddings=True, 1024-dim, multilingual |
| LLM | Qwen3-235B-A22B-Instruct-2507-tput | Together AI, Apache 2.0, non-thinking |
| Index | normalized dense tensor | GPU/MPS HBM-resident, per-project disk persist |

---

## 2. 검증된 성과

| 지표 | 값 | 조건 |
|---|---|---|
| 답변 정확도 | 97% | 100 query eval, v6 prompt, top_k=10 |
| Hallucination | 0% | 동일 eval |
| False refusal | 2% | 동일 eval |
| Brain recall latency | 10.45 ms | 영어 위키 6.4M articles, H100 |
| 의미 집중도 | 0.9809 | — |
| Scale (B200 검증) | 6.4M에서 sub-ms 배치 retrieval | 합성 벡터, dim=1024, `experiments/03` |

<!-- TODO: 외부 claim 시 — Faiss 배수는 GPU-Brain vs CPU-Faiss 비교라 부적합. 합성벡터 수치와 실제 정확도(97%/0.9809)는 별개 측정. -->

---

## 3. 디렉토리 구조

```
brain-poc/  (branch d4-ui)
  src/
    brain.py             - Modern Hopfield retrieval + save/load (persistence 핵심)
    orchestrator.py      - query -> Brain retrieve -> prompt -> LLM -> dual answer
    llm_client.py        - multi-provider adapter (Together + Anthropic)
    document_loader.py   - PDF/TXT -> Chunks
    db.py                - SQLite core (users/projects/files/chats/messages + allowed_emails)
    db_second_brain.py   - Second Brain 13-entity schema (892 lines)
    g1_extractor.py      - Second Brain G1 추출 (Step 2)
    pipeline.py          - APScheduler idle-chat poll + G1 trigger (Step 2)
    session_manager.py   - Session.brain, lazy reload 호환
    thinking_router.py   - Hybrid thinking 정책 (default ON + 자동 OFF)
    auth.py              - Google OAuth (Authlib + OIDC)
  app/
    server.py            - FastAPI main + SessionMiddleware
    api_d4.py            - REST endpoints (Project/Chat CRUD + persistence + dual mode)
    static/index.html    - Claude-style 3-view UI (markdown render + Pretendard)
  eval/
    queries_v1.json      - 100 baseline queries (5 카테고리 x 8 언어)
    run_eval.py          - 8-metric 자동 측정
  scripts/
    download_sherlock.py, load_corpus.py, test_e2e.py
    wiki/                - Wiki 6.4M 적재 (download/load/bench)
  experiments/
    01_associative_memory.py  - Hopfield 엔진 데모
    02_scale_benchmark.py     - scale 벤치 (mps/cpu)
    03_b200_benchmark.py      - GPU/CUDA scale 벤치 (B200 검증, dim=1024)
  data/projects/{pid}/brain_index/  - keys.pt + docs.json + meta.json (gitignored)
  logs/                              - 실행 로그 (gitignored)
```

<!-- TODO: src/app에 .bak.* 파일 다수 누적 (cleanup 후보, README 무관) -->

---

## 4. 아키텍처

### Dual answer mode
모든 query에 두 답변을 동시 표시:
- **📄 본 문서 기반** — Brain retrieval로 가져온 chunk에 근거 (0% 환각 narrative)
- **🌐 일반 지식 답변** — LLM general knowledge (reference docs inject)

### Persistence (launch blocker 해결됨)
파일 업로드 시 `BrainMemory`가 `data/projects/{pid}/brain_index/`에 자동 disk 저장. uvicorn 재시작/eviction 시 DB의 `brain_index_path`에서 lazy reload → 데이터 유실 없이 deploy/restart 자유.

### 인증
Google OAuth (multi-user). closed beta는 obscurity 기반 open access (allowlist는 `if False` bypass, 재활성화 1줄).

### Second Brain (layer 2, dev 중)
chat = thread + 그 chat으로부터 영구 진화한 모든 event의 합. G1 추출 → cluster → cross-link → outline → discovery. Step 1(13-entity DB schema) 완료, Step 2(G1 + pipeline) 진행 중.

---

## 5. 운영 인프라

| 항목 | 값 |
|---|---|
| Host | Lambda H100x1 (192.222.55.44, 80GB HBM3, CUDA 12.8) |
| Work dir | /home/ubuntu/brain-poc |
| Public | https://brain.umparumpa.com (Cloudflare Tunnel + brain-tunnel systemd) |
| LLM | Together AI (`LLM_PROVIDER=together`) |
| DB | SQLite `data/brain.db` (6 core + 13 Second Brain tables) |

---

## 6. 설치 / 실행

```bash
# 의존성 (venv)
.venv/bin/python -m pip install -r requirements.txt

# 환경변수
cp .env.example .env
# .env: TOGETHER_API_KEY, ANTHROPIC_API_KEY, GOOGLE_CLIENT_ID/SECRET, SESSION_SECRET

# 웹서버 (production, --reload 없이)
tmux new-session -d -s webui "uvicorn app.server:app --host 0.0.0.0 --port 8000 2>&1 | tee -a logs/uvicorn.log"
```

<!-- TODO: H100 실제 venv 경로/활성화 절차 확인해서 정확히 (.venv vs source .venv/bin/activate) -->

### 검증
```bash
.venv/bin/python eval/run_eval.py            # 100 query 8-metric
.venv/bin/python experiments/03_b200_benchmark.py   # GPU scale 벤치
```

---

## 7. 4-Phase 구조

| Phase | 상태 | 환경 |
|---|---|---|
| 0 사전 준비 | ✅ 종결 | 맥북 + Together API |
| 1 Closed Beta | 🔄 launch-ready | H100x1 + Together API + 라이브 |
| 2 자체 호스팅 검증 | 🔄 B200 포팅 검증 완료 (엔진/retrieval/scale) | B200 (single, Sydney) |
| 3 외부 데모 | ⏳ 대기 | — |

> Phase 2 노트: 단일 B200(192GB)로 Qwen3-235B fp8 로컬 serving 불가 (fp8는 H100 8장 필요). 단일 GPU 경로는 FP4/NVFP4(~132GB)만 가능 — 품질 회귀 재검증 필요한 별도 과제.

상세: Notion "Brain 운영 4-Phase Plan v1" / "Master Reference v3".

---

## 8. 알려진 함정 (개발 시 주의)

- **브랜치**: 실서비스 = `d4-ui`. `main`은 무관한 옛 코드.
- **torch 보호**: GPU 환경에서 `sentence-transformers` 설치 시 `--no-deps`로 torch 재설치 차단 (cu 버전 깨짐 방지).
- **uvicorn `--reload` 금지** (production): env load fragile + 코드 변경 시 race. 수동 restart.
- **heredoc + Python escape**: f-string 내 backslash, `\n` 이중 escape, re.sub replacement 주의 (Notion changelog v3.14 함정 ledger 참조).
- **Together Qwen3 non-thinking**: thinking variant 아님 (운영 모델 확정).
