# 30-min Network Break — 2026-04-24T23:13:16+00:00

## 상태
- bitsandbytes 0.49.2 설치 완료
- bf16 70B download PID: 10211, log: /home/ubuntu/xhbm/logs/bf16_download_20260424_231316.log
- smoke_02_bnb script 작성 완료: ~/xhbm/scripts/phase_c_smoke_02_bnb_baseline.py
- 서버 $3.29/hr 지속 과금 (30min idle ≈ $1.65, 재구축보다 쌈)

## 복귀 시 1줄
ssh xhbm 'bash -s' << BASH
ps -p $(cat ~/xhbm/logs/bf16_download.pid) 2>/dev/null && echo "STILL DOWNLOADING" || echo "DOWNLOAD DONE/DIED"
du -sh ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3.1-70B-Instruct 2>/dev/null
tail -n 10 /home/ubuntu/xhbm/logs/bf16_download_20260424_231316.log
BASH

## Download 완료됐으면 바로 실행
nohup python3 ~/xhbm/scripts/phase_c_smoke_02_bnb_baseline.py \
  > ~/xhbm/logs/smoke_02_bnb_$(date +%Y%m%d_%H%M%S).log 2>&1 &
