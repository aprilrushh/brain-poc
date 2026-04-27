# Phase C 수면 복귀 체크리스트 (2026-04-24 16:00 KST 작성)

## 한 줄 요약
FP8 H100 단일에서 실행 불가 확정 (compressed-tensors native kernel 부재).
AWQ INT4 피벗 결정만 남음. 결정되면 즉시 진행 가능한 상태.

## 복귀 순서
1. `tail -30 ~/xhbm/logs/heartbeat_*.log` — 수면 중 GPU/디스크 이상 여부
2. `nvidia-smi` — GPU 살아있는지
3. `ps aux | grep -E "python|hf|nohup" | grep -v grep` — 살아있는 프로세스 확인
4. Claude에게 피벗 결정 통보:
   - "AWQ OK, 피벗 진행" → autoawq 설치 + 40GB 다운로드 + smoke_02 작성
   - "GPTQ로" → 같은 크기 다른 repo
   - "vLLM으로" → 엔진 재설계 (시간 많이 듦)
   - "2x H100으로 fp8 유지" → xhbm-launch TARGETS 재설정 필요

## 서버 기본
- IP: 209.20.157.36  |  alias: `ssh xhbm`
- 과금: $3.29/hr, launch 2026-04-24 15:00 근처
- 디스크 여유: 약 830GB (FP8 두 버전 136GB + 환경 5GB 사용)

## Notion 핸드오프
- 🛌 Phase C 중단 (이 세션): https://www.notion.so/34cc78cb12ce811ea770c0e21b6d57f6
- 📊 §4 업데이트: https://www.notion.so/34cc78cb12ce813c9b63ea21e225519f
- 🎯 North Star: https://www.notion.so/34bc78cb12ce81ddbaebe1a3a62ee2b3

## 만약 Claude 세션이 끊겼으면
위 4개 Notion 페이지 순서대로 fetch하라고 하면 새 Claude가 맥락 복원.
