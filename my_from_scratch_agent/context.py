def prepare_context(system_prompt: str, memory: str, history: str, task: str):
    return f"{system_prompt}\n{memory}\n{history}\n{task}"
