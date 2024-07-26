# !pip install -U accelerate
# !pip install -U transformers
# !pip install datasets
# !pip install evaluate
# !pip install mlflow
# !pip install transformers[torch]
# !pip install jiwer
# !pip install nlptutti
# !huggingface-cli login --token token
'''
데이터셋을 허깅페이스로 업로드하는 코드에요. GPU로 작동해요.
'''
import os
import json
from pydub import AudioSegment
from tqdm import tqdm
import re
from datasets import Audio, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pandas as pd
import shutil

import torch
import torchaudio
import torchaudio.transforms as transforms

# 사용자 지정 변수를 설정해요.

set_num = 14                                                                       # 데이터셋 번호
token = "hf_"                                   # 허깅페이스 토큰
CACHE_DIR = '/mnt/a/maxseats/.cache_' + str(set_num)                              # 허깅페이스 캐시 저장소 지정
dataset_name = "maxseats/aihub-464-preprocessed-680GB-set-" + str(set_num)        # 허깅페이스에 올라갈 데이터셋 이름
model_name = "SungBeom/whisper-small-ko"                                          # 대상 모델 / "openai/whisper-base"
batch_size = 500                                                                 # 배치사이즈 지정, 8000이면 에러 발생

json_path = '/mnt/a/maxseats/mp3_dataset.json'                                    # 생성한 json 데이터셋 위치

print('현재 데이터셋 : ', 'set_', set_num)


# FeatureExtractor 클래스 정의
class FeatureExtractor:
    def __init__(self, sample_rate=16000, n_mels=64):
        """
        FeatureExtractor 클래스의 생성자입니다.

        Args:
            sample_rate (int, optional): 오디오 샘플링 속도입니다. 기본값은 16000입니다.
            n_mels (int, optional): Mel 스펙트로그램의 빈 수입니다. 기본값은 64입니다.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.mel_spectrogram_transform = transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels
        )
        self.amplitude_to_db = transforms.AmplitudeToDB()
    
    def to_device(self, device):
        """
        Mel 스펙트로그램 변환에 사용되는 텐서들을 지정된 디바이스로 이동시킵니다.

        Args:
            device (torch.device): 이동시킬 디바이스입니다.
        """
        self.mel_spectrogram_transform.spectrogram.window = self.mel_spectrogram_transform.spectrogram.window.to(device)
        self.mel_spectrogram_transform.mel_scale.fb = self.mel_spectrogram_transform.mel_scale.fb.to(device)
        
    def __call__(self, audio_tensor):
        """
        주어진 오디오 텐서를 Mel 스펙트로그램으로 변환합니다.

        Args:
            audio_tensor (torch.Tensor): 변환할 오디오 텐서입니다.

        Returns:
            torch.Tensor: 변환된 log-Mel 스펙트로그램 텐서입니다.
        """
        # Mel 스펙트로그램 변환
        mel_spectrogram = self.mel_spectrogram_transform(audio_tensor)
        # log-Mel 스펙트로그램 변환
        log_mel_spectrogram = self.amplitude_to_db(mel_spectrogram)
        return log_mel_spectrogram


def prepare_dataset(batch):
    """
    주어진 배치를 처리하여 데이터셋을 준비하는 함수입니다.

    Args:
        batch (dict): 처리할 배치 데이터의 딕셔너리. 다음 키를 포함해야 합니다:
            - "audio" (dict): 오디오 데이터를 포함하는 딕셔너리. 다음 키를 포함해야 합니다:
                - "array" (ndarray): 오디오 데이터의 배열.
            - "labels" (list): 레이블 데이터의 리스트.

    Returns:
        dict: "input_features"와 "labels" 키만 포함하는 새로운 딕셔너리. "input_features"는 로그 멜 스펙트로그램의 첫 번째 프레임을 CPU로 이동한 값입니다.
    """
    
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # GPU로 오디오 데이터를 텐서로 변환하고 float32로 캐스팅
    audio_tensor = torch.tensor(audio["array"], dtype=torch.float32).to(device=torch.device("cuda"))

    # 예시 feature_extractor 인스턴스 (GPU로 전처리 수행)
    feature_extractor = FeatureExtractor()
    feature_extractor.to_device(audio_tensor.device)

    # input audio array로부터 log-Mel spectrogram 변환
    log_mel_spectrogram = feature_extractor(audio_tensor.unsqueeze(0))

    # 첫 번째 스펙트로그램 프레임을 CPU로 이동
    batch["input_features"] = log_mel_spectrogram[0].cpu()

    # 'input_features'와 'labels'만 포함한 새로운 딕셔너리 생성
    return {"input_features": batch["input_features"], "labels": batch["labels"]}


# JSON 파일에서 데이터 로드
def load_dataset(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# 파일 경로 참조해서 오디오, set_num 데이터 정답 라벨 불러오기
def getLabels(json_path, set_num):
    """
    주어진 JSON 파일에서 set_num에 해당하는 데이터를 필터링하여 데이터프레임으로 반환합니다.
    
    Parameters:
        json_path (str): JSON 파일의 경로
        set_num (int): 필터링할 데이터의 set 번호
    
    Returns:
        pandas.DataFrame: 필터링된 데이터의 데이터프레임
    """
    
    # JSON 파일 로드
    json_dataset = load_dataset(json_path)
    
    set_identifier = 'set_' + str(set_num) + '/'
    
    # "audio" 경로에 set_identifier가 포함된 데이터만 필터링
    filtered_data = [item for item in json_dataset if set_identifier in item['audio']]

    return pd.DataFrame(filtered_data)


# Sampling rate 16,000khz 전처리 + 라벨 전처리를 통해 데이터셋 생성
def df_transform(batch_size, prepare_dataset):
    """
    주어진 데이터프레임을 처리하여 데이터셋으로 변환하는 함수입니다.

    Args:
        batch_size (int): 배치 크기입니다.
        prepare_dataset (Callable): 데이터셋을 준비하는 함수입니다.

    Returns:
        Dataset: 변환된 전체 데이터셋입니다.
    """
    batches = []
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        ds = Dataset.from_dict(
            {"audio": [path for path in batch_df["audio"]],
             "labels": [transcript for transcript in batch_df["transcripts"]]}
        ).cast_column("audio", Audio(sampling_rate=16000))

        batch_datasets = DatasetDict({"batch": ds})
        batch_datasets = batch_datasets.map(prepare_dataset, num_proc=1)
        batch_datasets.save_to_disk(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
        batches.append(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
        print(f"Processed and saved batch {i//batch_size}")

    # 모든 배치 데이터셋 로드, 병합
    loaded_batches = [load_from_disk(path) for path in batches]
    full_dataset = concatenate_datasets([batch['batch'] for batch in loaded_batches])

    return full_dataset

# 데이터셋을 훈련 데이터와 테스트 데이터, 밸리데이션 데이터로 분할
def make_dataset(full_dataset):
    """
    데이터셋을 생성하는 함수입니다.

    Parameters:
        full_dataset (Dataset): 전체 데이터셋

    Returns:
        datasets (DatasetDict): 학습, 테스트, 검증 데이터셋을 포함한 데이터셋 딕셔너리
    """
    train_testvalid = full_dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    datasets = DatasetDict(
        {"train": train_testvalid["train"],
         "test": test_valid["test"],
         "valid": test_valid["train"]}
    )
    return datasets

# 허깅페이스 로그인 후, 최종 데이터셋을 업로드
def upload_huggingface(dataset_name, datasets, token):
    """
    Hugging Face 데이터셋을 업로드하는 함수입니다.
    
    Parameters:
        dataset_name (str): 업로드할 데이터셋의 이름입니다.
        datasets (object): Hugging Face의 datasets 모듈 객체입니다.
        token (str): Hugging Face API 토큰입니다.
        
    Returns:
        None
    """
    while True:
        
        if token =="exit":
            break
        
        try:
            datasets.push_to_hub(dataset_name, token=token)
            print(f"Dataset {dataset_name} pushed to hub successfully. 넘나 축하.")
            break
        except Exception as e:
            print(f"Failed to push dataset: {e}")
            token = input("Please enter your Hugging Face API token: ")

for set_num in range(13, 69):  # 13부터 68까지의 데이터셋 처리 후 업로드

    CACHE_DIR = '/mnt/a/maxseats/.cache_' + str(set_num)                              # 허깅페이스 캐시 저장소 지정
    dataset_name = "maxseats/aihub-464-preprocessed-680GB-set-" + str(set_num)        # 허깅페이스에 올라갈 데이터셋 이름
    print('현재 데이터셋 : ', 'set_', set_num)
    
    # 캐시 디렉토리 설정
    os.environ['HF_HOME'] = CACHE_DIR
    os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
    tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe", cache_dir=CACHE_DIR)
    
    
    df = getLabels(json_path, set_num)
    print("len(df) : ", len(df))
    
    full_dataset = df_transform(batch_size, prepare_dataset)
    datasets = make_dataset(full_dataset)
    
    
    
    upload_huggingface(dataset_name, datasets, token)
    
    # 캐시 디렉토리 삭제
    shutil.rmtree(CACHE_DIR)
    print("len(df) : ", len(df))
    print(f"Deleted cache directory: {CACHE_DIR}")