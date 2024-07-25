from functools import partial
from typing import Dict

from transformers import Seq2SeqTrainer
from metrics import compute_metrics


def create_trainer(components: Dict) -> Seq2SeqTrainer:
    """
    Seq2Seq 모델을 훈련하기 위한 Seq2SeqTrainer 객체를 생성합니다.

    Args:
        components (Dict): 훈련자를 생성하는 데 필요한 구성 요소를 포함하는 딕셔너리입니다.
            - tokenizer: 입력 데이터를 토큰화하는 데 사용되는 토크나이저입니다.
            - metric: 모델의 성능을 평가하는 데 사용되는 메트릭입니다.
            - training_args: 훈련자를 구성하는 훈련 인자입니다.
            - model: 훈련될 Seq2Seq 모델입니다.
            - preprocessed_dataset: 전처리된 훈련 및 검증 데이터셋을 포함하는 딕셔너리입니다.
            - data_collator: 입력 데이터를 배치 처리하는 데 사용되는 데이터 콜레이터입니다.
            - processor: 특성 추출에 사용되는 프로세서입니다.

    Returns:
        Seq2SeqTrainer: 생성된 Seq2SeqTrainer 객체입니다.
    """
    tokenizer = components["tokenizer"]
    metric = components["metric"]
    compute_metrics_fn = partial(compute_metrics, tokenizer=tokenizer, metric=metric)

    trainer = Seq2SeqTrainer(
        args=components["training_args"],
        model=components["model"],
        train_dataset=components["preprocessed_dataset"]["train"],
        eval_dataset=components["preprocessed_dataset"]["valid"],
        data_collator=components["data_collator"],
        compute_metrics=compute_metrics_fn,
        tokenizer=components["processor"].feature_extractor,
    )

    return trainer


if __name__ == "__main__":
    import math
    from transformers import Seq2SeqTrainer
    from config.config_manager import get_config, get_components

    config = get_config()  # yaml 파일과 argparse를 통해 받은 args를 합친 config 불러오기
    components = get_components(config)  # model, dataset, trainig_arguments, ... 등을 불러오기

    # (test용)valid dataset의 10%만 사용
    if config["test"]:
        valid_dataset = components["preprocessed_dataset"]["valid"]
        valid_dataset = valid_dataset.select(
            range(math.ceil(len(components["preprocessed_dataset"]) * 0.1))
        )
        components["preprocessed_dataset"]["valid"] = valid_dataset

    trainer = create_trainer(components)
    trainer.train()
    metrics = trainer.evaluate()

    print(metrics)
