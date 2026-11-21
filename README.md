# noise_reduction

PyTorch 기반 CNN으로 잡음이 섞인 음성을 복원하는 챌린지 프로젝트입니다.

## 구성

- `deep_learning_noise_reduction.ipynb`: 환경 확인, 데이터 로딩, CNN 학습·검증, 결과 비교와 평가
- `outputs/best_mask_cnn.pt`: 검증 PESQ-WB 기준 최고 가중치
- `outputs/previews/<run_id>/epoch_*.pt`: 10·20·40·80 epoch 비교용 가중치
- `outputs/training_history.csv`: epoch별 학습 기록
- `outputs/training_summary.json`: 학습 설정과 최고 epoch 요약
- `outputs/validation_per_file.csv`, `outputs/validation_summary.csv`: 검증별·평균 PESQ-WB 및 SNR

원본 음성 데이터는 개인 음성이 포함될 수 있어 이 저장소에 포함하지 않습니다. 데이터를 별도로 준비해 주세요.

## 환경 및 데이터

노트북은 Python 3.12와 PyTorch, NumPy, pandas, SciPy, SoundFile, Matplotlib, tqdm, PESQ 0.0.4를 사용합니다. 첫 환경 셀은 현재 커널에 없는 주요 패키지를 설치하고 PESQ를 빌드합니다.

macOS에서 PESQ 빌드가 Xcode 라이선스로 실패하면 Terminal에서 `sudo xcodebuild -license`를 실행해 동의한 뒤 노트북의 첫 환경 셀을 다시 실행하세요. Windows Conda 사용자는 노트북의 1절 안내에 따라 `pesq`를 설치할 수 있습니다.

데이터 폴더는 `train/` 및 `validation/` 디렉터리와 각 `samples.csv`, `clean/`, `noisy/` WAV 파일을 포함해야 합니다. 저장소 폴더 아래 기본 이름인 `deep_learning_noise_reduction_challenge/`로 두거나, 노트북의 환경 설정 셀에서 `LOCAL_FOLDER`를 데이터 루트로 지정하세요. WAV는 16 kHz 모노, 3초입니다.

## 실행 순서

1. VS Code Jupyter에서 이 노트북을 열고 준비한 Python 커널을 선택합니다.
2. 위에서 아래로 실행해 환경과 데이터 경로를 확인하고 모델을 정의합니다.
3. 학습하려면 4.1절 함수 셀 다음 4.2절 학습 셀을 실행합니다. 기본 설정은 매 5 epoch마다 전체 PESQ를 계산하고, 프리뷰 epoch에서는 항상 계산합니다. 매 epoch 검증 손실은 기록됩니다.
4. 저장된 모델 비교는 4.4절과 5절에서, 전체 검증 PESQ-WB 및 SNR 평가는 6절에서 실행합니다.

학습이 끝나면 최고 체크포인트가 `outputs/best_mask_cnn.pt`에 저장됩니다. 이번 저장 결과는 180 epoch에서 조기 종료했고, 최고 epoch는 170, 검증 평균 PESQ-WB는 1.7322입니다. 평가 간격 사이의 `validation_pesq_wb` 빈 값은 해당 epoch에서 전체 PESQ 계산을 생략했다는 뜻입니다.

## 데이터 공개 주의

`train/`, `validation/`, 로컬 파형 캐시, 미리보기 WAV는 `.gitignore`에서 제외합니다. epoch 비교용 `.pt` 체크포인트는 포함하지만, 원본 음성 WAV나 음성이 포함된 오디오 결과를 공개 저장소에 추가하지 마세요.
