# !pip install -U accelerate
# !pip install -U transformers
# !pip install datasets
# !pip install evaluate
# !pip install mlflow
# !pip install transformers[torch]
# !pip install jiwer
# !pip install nlptutti


import os
import json
from pydub import AudioSegment
from tqdm import tqdm
import re
from datasets import Audio, Dataset, DatasetDict, load_from_disk, concatenate_datasets
from transformers import WhisperFeatureExtractor, WhisperTokenizer
import pandas as pd
import shutil

# 사용자 지정 변수를 설정해요.

output_dir = '/mnt/a/maxseats/(주의-원본)mp3_clips/set_10'           # 가공된 데이터셋이 저장된 폴더
token = "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"                     # 허깅페이스 토큰
CACHE_DIR = './.cache'                                              # 허깅페이스 캐시 저장소 지정
dataset_name = "maxseats/aihub-464-preprocessed-680GB-set-10"        # 허깅페이스에 올라갈 데이터셋 이름
model_name = "SungBeom/whisper-small-ko"                            # 대상 모델 / "openai/whisper-base"


batch_size = 2000   # 배치사이즈 지정, 8000이면 에러 발생
os.environ['HF_DATASETS_CACHE'] = CACHE_DIR

'''
가공된 mp3, txt 데이터를 학습 가능한 허깅페이스 데이터셋 형태로 변환해요.
'''



def exclude_json_files(file_names: list) -> list:
    # .json으로 끝나는 원소 제거
    return [file_name for file_name in file_names if not file_name.endswith('.json')]


def get_label_list(directory):
    # 빈 리스트 생성
    label_files = []

    # 디렉토리 내 파일 목록 불러오기
    for filename in os.listdir(directory):
        # 파일 이름이 '.txt'로 끝나는지 확인
        if filename.endswith('.txt'):
            label_files.append(os.path.join(directory, filename))

    return label_files


def get_audio_list(directory):
    # 빈 리스트 생성
    audio_files = []

    # 디렉토리 내 파일 목록 불러오기
    for filename in os.listdir(directory):
        # 파일 이름이 '.wav'나 '.mp3'로 끝나는지 확인
        if filename.endswith('.wav') or filename.endswith('mp3'):
            audio_files.append(os.path.join(directory, filename))

    return audio_files

def prepare_dataset(batch):
    # 오디오 파일을 16kHz로 로드
    audio = batch["audio"]

    # input audio array로부터 log-Mel spectrogram 변환
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # target text를 label ids로 변환
    batch["labels"] = tokenizer(batch["transcripts"]).input_ids
    
    # 'audio'와 'transcripts' 컬럼 제거
    # del batch["audio"]
    # del batch["transcripts"]
    
    # 'input_features'와 'labels'만 포함한 새로운 딕셔너리 생성
    return {"input_features": batch["input_features"], "labels": batch["labels"]}


# 캐시 디렉토리 설정
os.environ['HF_HOME'] = CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = CACHE_DIR
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name, cache_dir=CACHE_DIR)
tokenizer = WhisperTokenizer.from_pretrained(model_name, language="Korean", task="transcribe", cache_dir=CACHE_DIR)

label_data = get_label_list(output_dir)
audio_data = get_audio_list(output_dir)



def getLabels():


# 파일 경로 참조해서 정답 라벨들 불러오기
transcript_list = []
for label in tqdm(label_data):
    with open(label, 'r', encoding='UTF8') as f:
        line = f.readline()
        transcript_list.append(line)

df = pd.DataFrame(data=transcript_list, columns = ["transcript"]) # 정답 label
df['audio_data'] = audio_data # 오디오 파일 경로

# 오디오 파일 경로를 dict의 "audio" 키의 value로 넣고 이를 데이터셋으로 변환
# 이때, Whisper가 요구하는 사양대로 Sampling rate는 16,000으로 설정한다.
batches = []
print("len(df) : ", len(df))
for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
    batch_df = df.iloc[i:i+batch_size]
    ds = Dataset.from_dict(
        {"audio": [path for path in batch_df["audio_data"]],
         "transcripts": [transcript for transcript in batch_df["transcript"]]}
    ).cast_column("audio", Audio(sampling_rate=16000))
    
    batch_datasets = DatasetDict({"batch": ds})
    batch_datasets = batch_datasets.map(prepare_dataset, num_proc=1)
    batch_datasets.save_to_disk(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
    batches.append(os.path.join(CACHE_DIR, f'batch_{i//batch_size}'))
    print(f"Processed and saved batch {i//batch_size}")

# 모든 배치 데이터셋 로드, 병합
loaded_batches = [load_from_disk(path) for path in batches]
full_dataset = concatenate_datasets([batch['batch'] for batch in loaded_batches])

# 데이터셋을 훈련 데이터와 테스트 데이터, 밸리데이션 데이터로 분할
train_testvalid = full_dataset.train_test_split(test_size=0.2)
test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
datasets = DatasetDict(
    {"train": train_testvalid["train"],
     "test": test_valid["test"],
     "valid": test_valid["train"]}
)

# # 열 제거 전 데이터셋 크기 확인
# print(f"Dataset sizes before column removal: Train: {len(datasets['train'])}, Test: {len(datasets['test'])}, Valid: {len(datasets['valid'])}")

# datasets = datasets.remove_columns(['audio', 'transcripts'])  # 불필요한 부분 제거

# # 열 제거 후 데이터셋 크기 확인
# print(f"Dataset sizes after column removal: Train: {len(datasets['train'])}, Test: {len(datasets['test'])}, Valid: {len(datasets['valid'])}")

# #datasets = datasets.remove_columns(['audio', 'transcripts']) # 불필요한 부분 제거


'''
허깅페이스 로그인 후, 최종 데이터셋을 업로드해요.
'''

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

# 캐시 디렉토리 삭제
shutil.rmtree(CACHE_DIR)
print(f"Deleted cache directory: {CACHE_DIR}")