import os
import json
from pydub import AudioSegment
from tqdm import tqdm
import re
from datasets import Audio, Dataset, DatasetDict, load_dataset, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pandas as pd
import shutil

# 사용자 지정 변수를 설정해요.

# DATA_DIR = '/mnt/a/maxseats/(주의-원본-680GB)주요 영역별 회의 음성인식 데이터' # 데이터셋이 저장된 폴더
DATA_DIR = '/mnt/a/maxseats/(주의-원본)split_files/set_0'  # 첫 10GB 테스트

# 원천, 라벨링 데이터 폴더 지정
json_base_dir = DATA_DIR
audio_base_dir = DATA_DIR
output_dir = '/mnt/a/maxseats/(주의-원본)clips_set_0'                     # 가공된 데이터셋이 저장될 폴더
token = "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"                     # 허깅페이스 토큰
CACHE_DIR = '/mnt/a/maxseats/.cache'                                # 허깅페이스 캐시 저장소 지정
dataset_name = "maxseats/aihub-464-preprocessed-680GB-set-0-2"              # 허깅페이스에 올라갈 데이터셋 이름
model_name = "SungBeom/whisper-small-ko"                            # 대상 모델 / "openai/whisper-base"


batch_size = 5500   # 배치사이즈 지정, 8000이면 에러 발생
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR
'''
데이터셋 경로를 지정해서
하나의 폴더에 mp3, txt 파일로 추출해요. (clips_set_i 폴더)
추출 과정에서 원본 파일은 자동으로 삭제돼요. (저장공간 절약을 위해)
'''




# 캐시 디렉토리 생성 (필요한 경우)
os.makedirs(CACHE_DIR, exist_ok=True)

original_dataset_name = 'maxseats/aihub-464-preprocessed-680GB-set-0'
new_dataset_name = 'maxseats/aihub-464-preprocessed-680GB-set-0-2'
# 허깅페이스 데이터셋 로드
datasets = load_dataset(original_dataset_name, cache_dir=CACHE_DIR)

# 불필요한 컬럼 제거
datasets = datasets.remove_columns(['audio', 'transcripts'])
print('-'*48)
print(type(datasets))
print(datasets)
print('-'*48)

# 최종 데이터셋을 허깅페이스에 업로드
while True:
    if token == "exit":
        break
    
    try:
        datasets.push_to_hub(new_dataset_name, token=token)
        print(f"Dataset {new_dataset_name} pushed to hub successfully.")
        break
    except Exception as e:
        print(f"Failed to push dataset: {e}")
        token = input("Please enter your Hugging Face API token: ")
