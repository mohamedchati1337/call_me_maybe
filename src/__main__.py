from llm_sdk import Small_LLM_Model  # type: ignore[attr-defined]
from src.generation import constrained_decoding, generate_params
from src.models_and_utils import FunctionDefinition, TestPrompt, args
import json
import os
import sys
from typing import Any, Dict, List
from pydantic import ValidationError
sys.path.append("llm_sdk")


def main() -> None:
    """Run the main benchmark evaluation pipeline."""
    ar = args()

    try:
        with open(ar.input, "r") as file:
            data = json.load(file)
    except Exception as e:
        print(f"Error opening or reading input file: {e}")
        sys.exit(1)

    validated_prompts: List[TestPrompt] = []
    for item in data:
        try:
            validated_prompts.append(TestPrompt(**item))
        except ValidationError as e:
            error_msg = e.errors()[0]["msg"]
            print(f"Error parsing prompt JSON: {error_msg}")
            sys.exit(1)

    try:
        with open(ar.functions_definition, "r") as file:
            functions_data = json.load(file)
    except Exception as e:
        print(f"Error opening functions definition file: {e}")
        sys.exit(1)

    functions_data.append({
        "name": "fn_unknown",
        "description": (
            "Fallback option when the user request "
            "does not match any other function"
        ),
        "parameters": {},
        "returns": {}
    })

    validated_functions: List[FunctionDefinition] = []
    for item in functions_data:
        try:
            validated_functions.append(FunctionDefinition(**item))
        except ValidationError as e:
            error_msg = e.errors()[0]["msg"]
            print(f"Error parsing prompt JSON: {error_msg}")
            sys.exit(1)

    model = Small_LLM_Model()
    final_output_list: List[Dict[str, Any]] = []

    for i in range(len(validated_prompts)):
        sample_prompt: str = validated_prompts[i].prompt
        print(f"\n[Processing {i+1}/11] Prompt: {sample_prompt}")

        fjs: str = json.dumps(functions_data, indent=2)
        av_func: str = "Available functions"

        guided_prompt: str = f"{av_func} {fjs}\nRequest: {sample_prompt}\n"
        guided_prompt += 'JSON Output: {"function_name": "'

        func_name: str = constrained_decoding(
            model=model,
            prompt_text=guided_prompt,
            functions_data=validated_functions,
            max_new_tokens=20,
        )
        print(f"-> Captured Function Name: {func_name}")

        param_prompt: str = guided_prompt + func_name + '", "parameters": {'
        raw_params: str = generate_params(
            model=model,
            prompt_text=param_prompt,
            max_new_tokens=40,
        )
        print(f"-> Captured Raw Parameters: {raw_params}")

        fixed_raw_params: str = raw_params.replace("\\", "\\\\")

        parsed_parameters: Dict[str, Any] = {}
        try:
            parsed_parameters = json.loads(fixed_raw_params)
        except Exception:
            parsed_parameters = {}

        for f in validated_functions:
            if f.name == func_name:
                for p in f.parameters:
                    if f.parameters[p]['type'] == 'number':
                        parsed_parameters[p] = float(parsed_parameters[p])

        result_object: Dict[str, Any] = {
            "prompt": sample_prompt,
            "fn_name": func_name,
            "args": parsed_parameters,
        }

        final_output_list.append(result_object)

    output_file_path: str = ar.output
    output_dir = os.path.dirname(output_file_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file_path, "w") as out_file:
        json.dump(final_output_list, out_file, indent=2)

    print(
        f"\n✅ All tasks completed! "
        f"Final file saved at: {output_file_path}"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
