def validate(item):
    return item["valid"] is True


def process_batch(batch):
    payloads = batch["payloads"]
    return [validate(item) for item in payload]
