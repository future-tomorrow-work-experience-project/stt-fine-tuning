def compute_metrics(pred, tokenizer, metric):
    """
    주어진 예측값과 레이블로부터 CER(Character Error Rate)를 계산하는 함수입니다.

    Args:
        pred (object): 예측값을 포함하는 객체입니다.
        tokenizer (object): 토크나이저 객체입니다.
        metric (object): CER 계산을 위한 메트릭 객체입니다.

    Returns:
        dict: CER 값을 담은 딕셔너리입니다.
    """
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad_token을 -100으로 치환
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # metrics 계산 시 special token들을 빼고 계산하도록 설정
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}