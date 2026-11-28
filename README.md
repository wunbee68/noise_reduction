# noise_reduction

PyTorch 기반 CNN으로 잡음이 섞인 음성을 복원하는 챌린지 프로젝트입니다. 예제는 16 kHz 모노, 3초 길이의 정렬된 noisy/clean 쌍을 사용합니다. 학습 1,296쌍, 검증 164쌍입니다.

## 0. 제출용 음성 복원

2026 신호 및 시스템 2차 챌린지 **5조** 제출 버전은 `challenge-2026-pesq-3.2112` 태그로 식별합니다. 런어스 질의응답 비밀글에는 팀명과 실제 제출 회차, 이 태그가 가리키는 버전 링크, 해당 회차의 AI 활용 내역서를 첨부합니다.

기본 제출 설정은 MP-SENet DNS 출력과 보정된 FastEnhancer DNS-L 출력을 80:20으로 혼합합니다. 배포 Train 1,296쌍으로 학습한 보정기는 FastEnhancer 분기에 강도 0.5로 적용합니다. 비교와 설정 선택에는 배포 Validation 164쌍을 사용했습니다.

### 설치 및 추론

Python 3.12, macOS ARM64 CPU에서 확인했습니다. 저장소 루트에서 다음 명령을 실행합니다.

```bash
git clone --branch challenge-2026-pesq-3.2112 https://github.com/wunbee68/noise_reduction.git
cd noise_reduction
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
mkdir -p input
# 배포받은 테스트 WAV를 input/에 배치
python infer.py
```

`infer.py`는 `input/` 아래의 WAV를 읽고 대응하는 출력 하나를 `output/` 바로 아래에 저장합니다. 파일명, 샘플링 주파수, 채널 수, 표본 수가 유지됩니다. 평가 입력은 16 kHz 모노, 48,000표본이고 출력은 PCM 16-bit WAV입니다. 기존 `output/`에 다른 이름의 WAV가 있으면 빈 폴더를 `--output-dir`로 지정하세요. 입력과 출력 경로를 바꾸려면 `--input-dir`와 `--output-dir`를 사용합니다.

추론 설정은 `submission.json`에 있습니다. `weights/mpsenet_dns.pt`, `weights/mpsenet_settings.json`, `weights/fastenhancer_dns_l.onnx`, `weights/fastenhancer_refiner_epoch08.pt`를 로컬에서 엽니다. 가장 큰 파일은 약 9.1 MB이며 일반 Git에 포함할 수 있습니다. 설치 후 추론에는 네트워크가 필요 없습니다.

기본 CPU 스레드 수는 2개입니다. 로컬 macOS에서 Validation 164개를 WAV로 저장하기까지 약 8분 21초가 걸렸습니다. 실행 시간은 장비에 따라 달라집니다.

### 모델과 학습 설정

1. MP-SENet은 입력 RMS를 정규화하고 Hann 창 400표본, hop 100으로 STFT를 수행합니다. 복소 스펙트럼에서 위상과 0.3제곱 압축 크기를 구합니다. 시간·주파수 Transformer 블록 4개를 통과한 특징으로 크기 마스크와 위상을 예측하고, 압축을 되돌린 뒤 iSTFT와 역정규화를 적용합니다.
2. FastEnhancer DNS-L은 16 kHz 파형을 100표본씩 처리하며 파일마다 상태를 초기화합니다. 마지막에 412표본의 지연을 제거하고 입력 길이를 복원합니다.
3. 보정 CNN은 noisy, 1차 복원, 두 파형의 차이에서 STFT 크기를 계산합니다. 6개의 dilated residual 블록을 사용해 1차 복원의 크기·위상과 잔차 비율을 보정합니다. 0.3제곱 압축 크기 L1과 압축 복소 스펙트럼 L1을 `1:0.5`로 합산해 AdamW로 학습합니다.
4. 보정 CNN은 동결된 DNS-B의 Train 출력을 사용했습니다. seed 42, 채널 96, batch 24, 초기 학습률 0.001을 사용했고, 2 epoch마다 Validation 전체의 PESQ-WB를 비교해 epoch 8 가중치를 골랐습니다. 같은 보정기를 DNS-L에 적용한 뒤 강도 0.5와 혼합 비율 0.2를 Validation에서 골랐습니다.

### Validation 재현

```bash
python -m pip install -r requirements-eval.txt
python evaluate.py --data-root ../deep_learning_noise_reduction_challenge \
    --examples-dir analysis_examples
```

`--predictions-dir output`으로 이미 저장된 WAV를 평가할 수 있습니다. PESQ는 16 kHz wideband로 계산합니다. SNR은 파일마다 `10 log10(clean 에너지 / 파형 오차 에너지)`로 계산합니다. 각 지표의 파일별 점수를 단순평균합니다. 누락, 추가 WAV, 형식 불일치 또는 PESQ 오류가 있으면 평가가 중단됩니다. 파일별·요약 CSV는 `outputs/submission_validation/`에 저장됩니다. 선택한 3개 사례의 PNG와 noisy/복원/clean WAV는 `analysis_examples/`에 저장되며 Git에서 제외됩니다.

| Validation 164개 | 평균 PESQ-WB | 평균 SNR (dB) |
| --- | ---: | ---: |
| Noisy 입력 | 1.3947 | 9.6341 |
| 예제 CNN, epoch 170 | 1.7322 | 13.5338 |
| MP-SENet VoiceBank+DEMAND | 1.7914 | 11.0003 |
| FastEnhancer DNS-B | 2.2900 | 15.1992 |
| FastEnhancer DNS-L | 2.4970 | 16.5550 |
| MP-SENet DNS | 3.1759 | 18.2313 |
| 최종 혼합 | **3.2112** | **18.2951** |

같은 80:20 비율의 미보정 혼합은 PESQ-WB 3.2113, SNR 18.2407 dB였습니다. PESQ-WB 차이가 약 0.0001인 두 조합 중 SNR이 0.0544 dB 높은 Train 보정 조합을 선택했습니다. 최종 조합은 Validation에서 Noisy 대비 모든 파일의 PESQ-WB가 올랐습니다. 변경 전후의 파일별 SNR은 CSV에서 확인할 수 있습니다.

스펙트로그램의 사이렌 사례에서는 약 1–2 kHz와 2.5–5 kHz 구간에 반복되던 주파수 궤적이 줄고, 저주파 음성 배음과 약 1.8초의 고주파 발화 성분이 남았습니다. 바람 사례에서는 약 2.0–2.4초의 저주파 배경이 줄었고 PESQ-WB는 2.06에서 2.30으로 높아졌으나 SNR은 15.00에서 8.14 dB로 낮아졌습니다. 낮은 입력 SNR에서도 차이가 컸습니다. 목표 SNR 0 dB인 15개는 평균 PESQ-WB 2.49, 목표 SNR 15 dB인 56개는 3.57이었습니다. Validation의 잡음 코드 61종은 모두 Train에도 있어, 히든 테스트의 미지 잡음에 대한 성능은 별도 평가가 필요합니다.

보정기에 대해 동일한 학습을 실행하려면 다음 명령을 사용합니다.

```bash
python train_refiner.py --data-root ../deep_learning_noise_reduction_challenge \
    --first-pass weights/fastenhancer_dns_b.onnx \
    --output-dir outputs/refiner_retrain --epochs 8
```

학습 및 평가에는 배포된 페어 WAV와 각 분할의 `samples.csv`가 필요합니다. 외부 WAV를 별도로 사용한 실험은 없었습니다. MP-SENet DNS 가중치는 저자의 DNS Challenge 2020 사전학습본이며, 논문은 −5~15 dB의 합성 혼합 3,000시간을 학습에 사용했다고 보고합니다. 비교용 VoiceBank+DEMAND 가중치는 저자가 28명 화자의 공식 학습 분할로 학습한 버전입니다. 구현과 두 체크포인트는 커밋 `89932cfe90d1dacb8e170e4a331d762462c21792`를 고정했습니다. 공개 가중치 호환을 위해 Transformer와 GRU의 입력 축 순서를 유지합니다. [논문](https://arxiv.org/html/2308.08926v2), [공식 체크포인트와 코드](https://github.com/yxlu-0102/MP-SENet/tree/89932cfe90d1dacb8e170e4a331d762462c21792)

FastEnhancer DNS-B와 DNS-L은 저자의 `onnx-dns-v1.0.0` 16 kHz 파형 모델입니다. 저자는 DNS Challenge 3 wideband 학습 자료를 사용하고 감정 발화와 가창을 제외했으며, 화자 `p232`, `p257`을 제외한 VCTK-0.92 음성을 추가했습니다. [학습 자료 명세](https://github.com/aask1357/fastenhancer#dns-challenge-16khz), [공식 ONNX 릴리스](https://github.com/aask1357/fastenhancer/releases/tag/onnx-dns-v1.0.0)

## 1. 개발 기반 및 환경

아래 1–8절은 기존 예제 CNN 실험의 재현 절차입니다. 개발·실행 기반은 VS Code Jupyter와 Python 3.12입니다. CNN 학습은 macOS, PyTorch 2.15.0, CPU에서 수행했습니다. 노트북의 환경 셀은 PyTorch, NumPy, pandas, SciPy, SoundFile, Matplotlib, tqdm을 확인하고 누락된 패키지를 설치합니다. PESQ-WB 평가는 `pesq==0.0.4`를 사용합니다.

macOS에서 PESQ 빌드가 Xcode 라이선스 오류로 실패하면 Terminal에서 `sudo xcodebuild -license`를 실행해 동의한 뒤 노트북의 첫 환경 셀을 다시 실행하세요. Windows Conda 커널 설치 방법은 노트북 1절을 참고하세요.

## 2. 데이터 준비

데이터셋은 이 저장소에 포함하지 않습니다. 개인 음성이 포함될 수 있으므로 원본 WAV를 공개 저장소에 추가하지 마세요. 데이터 루트에는 다음 구조가 필요합니다.

```text
deep_learning_noise_reduction_challenge/
├── train/
│   ├── samples.csv
│   ├── clean/*.wav
│   └── noisy/*.wav
└── validation/
    ├── samples.csv
    ├── clean/*.wav
    └── noisy/*.wav
```

WAV는 16 kHz 모노, 48,000표본(3초)이어야 합니다. 데이터 폴더를 저장소 루트 아래 기본 폴더명으로 두거나, 노트북 1절의 환경 설정 셀에서 `LOCAL_FOLDER`를 데이터 루트 경로로 지정하세요.

## 3. 실행 방법

1. VS Code에서 저장소를 열고 `deep_learning_noise_reduction.ipynb`를 Python 3.12 커널로 엽니다.
2. 1–3절을 위에서 아래로 실행해 패키지, 데이터 경로, clean/noisy 쌍, 모델 구조를 확인합니다.
3. 새로 학습하려면 4.1절의 함수 정의 셀 다음 4.2절 학습 셀을 실행합니다. Clean STFT 목표는 설정별로 `.waveform_cache/`에 저장되어 재사용됩니다.
4. 4.4절에서 epoch 10·20·40·80 결과를 비교하고, 5절에서 Wiener 필터와 CNN을 비교합니다.
5. 6절에서 검증 164개 전체 PESQ-WB와 SNR을 계산해 파일별·요약 CSV를 갱신합니다.

기본 설정은 매 epoch 손실을 검증하고, `PESQ_EVAL_INTERVAL=5`마다 전체 PESQ-WB를 계산합니다. `PREVIEW_EPOCHS`는 항상 평가합니다. 최고 가중치, 학습 기록, 학습 요약과 결과 CSV는 `outputs/`에 저장됩니다.

## 4. 적용 알고리즘

모델은 noisy 파형에 512표본 Hann 창, hop 128로 STFT를 적용합니다. 257개 주파수 빈의 `log1p(|STFT|)`를 1D CNN에 넣어 시간축별 마스크를 예측합니다. CNN은 Conv1d 채널 `(64, 64)`, 커널 `(5, 3, 1)`, ReLU 두 층과 Sigmoid 출력을 사용합니다. 예측 마스크를 noisy 복소 스펙트럼에 곱해 위상은 유지하고 크기를 조정한 뒤 inverse STFT로 파형을 복원합니다.

학습 손실은 복원 스펙트럼과 clean 스펙트럼 간 로그 크기 L1입니다. Adam으로 최적화하며, 전체 검증 PESQ-WB를 가중치 선택 기준, 학습률 조정 기준, 조기 종료 기준으로 사용합니다. Noisy 입력은 동일한 파일별 지표 비교의 기준선입니다.

## 5. 최종 학습 설정

| 설정 | 값 |
| --- | ---: |
| Seed | 42 |
| Sample rate | 16,000 Hz |
| STFT `n_fft` / `hop` | 512 / 128 |
| CNN hidden channels | `(64, 64)` |
| CNN kernel sizes | `(5, 3, 1)` |
| Batch size | 16 |
| Adam learning rate | 0.001 |
| Adam betas / weight decay | `(0.9, 0.999)` / 0.0 |
| Maximum / minimum epochs | 200 / 30 |
| PESQ evaluation interval | 5 epochs; preview epochs are always evaluated |
| Early-stop patience / min delta | 20 epochs / 0.001 |
| LR scheduler patience setting / factor / minimum LR | 5 epochs / 0.5 / 0.00001 |
| Preview epochs / sample index | `10, 20, 40, 80` / 138 |

PESQ를 5 epoch마다 평가하므로 `LR_PATIENCE=5`는 실제 `ReduceLROnPlateau.patience=1` 평가 간격으로 환산됩니다. 조기 종료의 patience는 전체 epoch 기준입니다.

## 6. Validation 결과

최종 실행은 180 epoch에서 조기 종료했고, 평가한 epoch 중 최고 가중치는 **epoch 170**입니다. 아래 값은 저장된 CNN 체크포인트로 검증 164개 전체를 다시 평가한 결과입니다.

| 비교 | 파일 수 | 평균 PESQ-WB | 평균 SNR (dB) |
| --- | ---: | ---: | ---: |
| Noisy 입력 | 164 | 1.3947 | 9.6341 |
| CNN 복원 | 164 | 1.7322 | 13.5338 |
| 개선량 (CNN − 입력) | 164 | +0.3375 | +3.8996 |

## 7. 기준 CNN 실험 파일

```text
.
├── .gitignore
├── README.md
├── deep_learning_noise_reduction.ipynb
├── figures/cnn_mask.png
└── outputs/
    ├── best_mask_cnn.pt
    ├── training_history.csv
    ├── training_summary.json
    ├── validation_per_file.csv
    ├── validation_summary.csv
    └── previews/<run_id>/epoch_{0010,0020,0040,0080}.pt
```

노트북은 GitHub에서 바로 열 수 있도록 실행 출력과 오디오 데이터를 제거한 사본입니다. 데이터 WAV, 로컬 STFT 캐시, 미리보기 WAV는 `.gitignore`에서 제외하며, 비교 재현용 epoch 가중치는 저장소에 포함합니다.

## 8. 참고 문헌 및 문서

- PyTorch, [`torch.stft`](https://docs.pytorch.org/docs/stable/generated/torch.stft.html) 및 [`torch.istft`](https://docs.pytorch.org/docs/stable/generated/torch.istft.html) — STFT/역 STFT 구현.
- PyTorch, [`torch.nn.Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) — CNN 시간축 합성곱 계층.
- PyTorch, [`ReduceLROnPlateau`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) — 검증 성능 기반 학습률 스케줄.
- ITU-T, [Recommendation P.862.2 (11/07)](https://www.itu.int/rec/T-REC-P.862.2/en), *Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs* — 이 프로젝트의 PESQ-WB 기준입니다. ITU 페이지는 이 권고가 2024년에 철회되었으며 현재 P.863 계열을 참조한다고 안내합니다.
- `pesq` Python package, [PyPI](https://pypi.org/project/pesq/) — PESQ-WB 계산 구현.
