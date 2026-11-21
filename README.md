# noise_reduction

PyTorch 기반 CNN으로 잡음이 섞인 음성을 복원하는 챌린지 프로젝트입니다. 예제는 16 kHz 모노, 3초 길이의 정렬된 noisy/clean 쌍을 사용합니다. 학습 1,296쌍, 검증 164쌍입니다.

## 개발 기반 및 환경

개발·실행 기반은 VS Code Jupyter와 Python 3.12입니다. 이번 학습은 macOS, PyTorch 2.15.0, CPU에서 수행했습니다. 노트북의 환경 셀은 PyTorch, NumPy, pandas, SciPy, SoundFile, Matplotlib, tqdm을 확인하고 누락된 패키지를 설치합니다. PESQ-WB 평가는 `pesq==0.0.4`를 사용합니다.

macOS에서 PESQ 빌드가 Xcode 라이선스 오류로 실패하면 Terminal에서 `sudo xcodebuild -license`를 실행해 동의한 뒤 노트북의 첫 환경 셀을 다시 실행하세요. Windows Conda 커널 설치 방법은 노트북 1절을 참고하세요.

## 데이터 준비

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

## 실행 방법

1. VS Code에서 저장소를 열고 `deep_learning_noise_reduction.ipynb`를 Python 3.12 커널로 엽니다.
2. 1–3절을 위에서 아래로 실행해 패키지, 데이터 경로, clean/noisy 쌍, 모델 구조를 확인합니다.
3. 새로 학습하려면 4.1절의 함수 정의 셀 다음 4.2절 학습 셀을 실행합니다. Clean STFT 목표는 설정별로 `.waveform_cache/`에 저장되어 재사용됩니다.
4. 4.4절에서 epoch 10·20·40·80 결과를 비교하고, 5절에서 Wiener 필터와 CNN을 비교합니다.
5. 6절에서 검증 164개 전체 PESQ-WB와 SNR을 계산해 파일별·요약 CSV를 갱신합니다.

기본 설정은 매 epoch 손실을 검증하고, `PESQ_EVAL_INTERVAL=5`마다 전체 PESQ-WB를 계산합니다. `PREVIEW_EPOCHS`는 항상 평가합니다. 최고 가중치, 학습 기록, 학습 요약과 결과 CSV는 `outputs/`에 저장됩니다.

## 적용 알고리즘

모델은 noisy 파형에 512표본 Hann 창, hop 128로 STFT를 적용합니다. 257개 주파수 빈의 `log1p(|STFT|)`를 1D CNN에 넣어 시간축별 마스크를 예측합니다. CNN은 Conv1d 채널 `(64, 64)`, 커널 `(5, 3, 1)`, ReLU 두 층과 Sigmoid 출력을 사용합니다. 예측 마스크를 noisy 복소 스펙트럼에 곱해 위상은 유지하고 크기를 조정한 뒤 inverse STFT로 파형을 복원합니다.

학습 손실은 복원 스펙트럼과 clean 스펙트럼 간 로그 크기 L1입니다. Adam으로 최적화하며, 전체 검증 PESQ-WB를 가중치 선택 기준, 학습률 조정 기준, 조기 종료 기준으로 사용합니다. Noisy 입력은 동일한 파일별 지표 비교의 기준선입니다.

## 최종 학습 설정

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

## Validation 결과

최종 실행은 180 epoch에서 조기 종료했고, 평가한 epoch 중 최고 가중치는 **epoch 170**입니다. 아래 값은 저장된 CNN 체크포인트로 검증 164개 전체를 다시 평가한 결과입니다.

| 비교 | 파일 수 | 평균 PESQ-WB | 평균 SNR (dB) |
| --- | ---: | ---: | ---: |
| Noisy 입력 | 164 | 1.3947 | 9.6341 |
| CNN 복원 | 164 | 1.7322 | 13.5338 |
| 개선량 (CNN − 입력) | 164 | +0.3375 | +3.8996 |

## 저장소 파일

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

## 참고 문헌 및 문서

- PyTorch, [`torch.stft`](https://docs.pytorch.org/docs/stable/generated/torch.stft.html) 및 [`torch.istft`](https://docs.pytorch.org/docs/stable/generated/torch.istft.html) — STFT/역 STFT 구현.
- PyTorch, [`torch.nn.Conv1d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html) — CNN 시간축 합성곱 계층.
- PyTorch, [`ReduceLROnPlateau`](https://docs.pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) — 검증 성능 기반 학습률 스케줄.
- ITU-T, [Recommendation P.862.2 (11/07)](https://www.itu.int/rec/T-REC-P.862.2/en), *Wideband extension to Recommendation P.862 for the assessment of wideband telephone networks and speech codecs* — 이 프로젝트의 PESQ-WB 기준입니다. ITU 페이지는 이 권고가 2024년에 철회되었으며 현재 P.863 계열을 참조한다고 안내합니다.
- `pesq` Python package, [PyPI](https://pypi.org/project/pesq/) — PESQ-WB 계산 구현.
