from typing import Any, List, Set
from src.models_and_utils import FunctionDefinition


def get_valid_next_tokens(
    current_generated: str, valid_names: List[str], model: Any
) -> Set[int]:
    """Isolate valid token IDs for the next generation step."""
    prefix_candidates: Set[int] = set()
    for name in valid_names:
        if name.startswith(current_generated):
            remaining_part = name[len(current_generated):]
            next_tok_ids: List[int] = model.encode(remaining_part).tolist()[0]
            if next_tok_ids:
                prefix_candidates.add(next_tok_ids[0])
    return prefix_candidates


def constrained_decoding(
    model: Any,
    prompt_text: str,
    functions_data: List[FunctionDefinition],
    max_new_tokens: int = 40,
) -> str:
    """Execute token generation restricted to valid function names."""
    input_ids = model.encode(prompt_text)
    input_ids_list: List[int] = list(input_ids[0])
    valid_names: List[str] = [f.name for f in functions_data]
    prompt_len: int = len(input_ids[0])
    respond: str = ""

    for _ in range(max_new_tokens):
        logits: List[float] = model.get_logits_from_input_ids(input_ids_list)
        current_generated: str = model.decode(input_ids_list[prompt_len:]) #" empty "

        if current_generated in valid_names:
            break

        final_allowed_tokens: Set[int] = get_valid_next_tokens(
            current_generated=current_generated,
            valid_names=valid_names,
            model=model,
        )

        # if current_generated == "":
        #     highest_score_token = logits.index(max(logits))
        #     if highest_score_token not in final_allowed_tokens:
        #         return "fn_None"

        # if not final_allowed_tokens:
        #     return "fn_None"

        constrained_logits: List[float] = [-float("inf")] * len(logits)
        for t_id in final_allowed_tokens:
            if t_id < len(logits):
                constrained_logits[t_id] = logits[t_id]

        next_token_id: int = constrained_logits.index(max(constrained_logits))
        input_ids_list.append(next_token_id)
        respond += model.decode(next_token_id)

    return respond.strip()


def generate_params(
    model: Any, prompt_text: str, max_new_tokens: int = 40
) -> str:
    """Perform free generation to extract the arguments JSON block."""
    input_ids = model.encode(prompt_text)
    input_ids_list: List[int] = list(input_ids[0])
    respond: str = ""

    for _ in range(max_new_tokens):
        logits: List[float] = model.get_logits_from_input_ids(input_ids_list)
        next_token_id: int = logits.index(max(logits))
        input_ids_list.append(next_token_id)

        token_str: str = model.decode(next_token_id)
        respond += token_str

        if "}" in respond:
            break

    if "}" in respond:
        respond = respond.split("}")[0]

    return "{" + respond.strip() + "}"
