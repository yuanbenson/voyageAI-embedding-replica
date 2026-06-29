def work_queue_key(prefix: str, model: str, workload: str) -> str:
    return f"{prefix}:work:embed:{model}:{workload}"


def result_queue_key(prefix: str, gateway_id: str) -> str:
    return f"{prefix}:results:{gateway_id}"
