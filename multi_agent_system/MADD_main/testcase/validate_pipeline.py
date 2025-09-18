"""
Modules for assessing the quality of the pipeline and individual agents
"""
import re
import pandas as pd

dict_for_map_many_func = {
    "alzheimer": "gen_mols_alzheimer",
    "dyslipidemia": "gen_mols_dyslipidemia",
    "lung cancer": "gen_mols_lung_cancer",
    "sclerosis": "gen_mols_multiple_sclerosis",
    "Drug_Resistance": "gen_mols_acquired_drug_resistance",
    "Parkinson": "gen_mols_parkinson",
    "nothing": "nothing",
}

dict_for_map_few_func = {
    "alzheimer": "Alzhmr",
    "dyslipidemia": "Dslpdm",
    "lung cancer": "Cnsr",
    "sclerosis": "Sklrz",
    "Drug_Resistance": "TBLET",
    "Parkinson": "Prkns",
}

def exctrac_mols_and_props(mols: str) -> list:
    pattern = r"\| ([A-Za-z0-9@+\-=#\[\]\(\)\\\/\.\*]+) \|"

    molecules = re.findall(pattern, mols)
    valid_molecules = [mol for mol in molecules if any(c.isalpha() for c in mol)]
    return valid_molecules


def check_total_answer(true_answers: list[str], total_answer: str) -> bool:
    for true_ans in true_answers:
        if true_ans == 'Molecules':
            continue
        if not (true_ans in total_answer):
            return False
    return True

def add_answers(answers: list, path: str):
    llm_res, tool_res, check, ts, bool_ts = answers
    pd.DataFrame({'llm_result': llm_res, 'tool_result': tool_res, 'check': check, 'ts': ts, 'bool_ts': bool_ts}).to_excel(path)

def validate_decompose(
    idx: int,
    decompose_lst: list,
    validation_path="./experiment3.xlsx",
) -> bool:
    lines = pd.read_excel(validation_path)
    columns = lines.columns
    lines = lines.values.tolist()

    num_tasks_true = len(lines[idx][0].split(","))
    lines[idx][2] = decompose_lst

    if len(decompose_lst) == num_tasks_true:
        lines[idx][3] = True

        pd.DataFrame(lines, columns=columns).to_excel(validation_path, index=False)
        return True
    else:
        lines[idx][3] = False
        pd.DataFrame(lines, columns=columns).to_excel(validation_path, index=False)
        return False


def validate_conductor(
    idx: int,
    func: dict,
    sub_task_number: int,
    path_total_val="multi_agents_system/testcase/experiment3_example.xlsx",
) -> bool:
    """
    Validate conductors agent answer. File must consist of next columns = 'case', 'content', 'decomposers_tasks', 'is_correct_context', 'task 1', 'task 2', 'task 3', 'task 4', 'task 5'

    Parameters
    ----------
    idx : int
        Number of line for validation
    func : dict
        Dict with function name and parameters
    sub_task_number : int
        Number of subtask (from decompose agent)

    Returns
    -------
    answer : bool
        Validation passed or not
    """
    lines = pd.read_excel(path_total_val)
    columns = lines.columns
    lines = lines.values.tolist()

    try:
        target_name = lines[idx][0].split(", ")[sub_task_number]
    except:
        target_name = "nothing"
    if isinstance(func, bool):
        return False

    # if call chat model for answer in free form (сos no such case exists in the file)
    if func["name"].replace(" ", "") == "make_answer_chat_model":
        lines[idx][4 + sub_task_number] = func["name"].replace(" ", "")
        pd.DataFrame(
            lines,
            columns=columns,
        ).to_excel(path_total_val, index=False)
        return False
    # if few function with a lot of args
    elif func["name"].replace(" ", "") == "gen_mols_all_case":
        if func["parameters"]["generation_type"] == dict_for_map_few_func[target_name]:
            lines[idx][4 + sub_task_number] = func["parameters"]["generation_type"]
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return True
        else:
            lines[idx][4 + sub_task_number] = func["parameters"]["generation_type"]
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return False
    # if many function with a few args
    else:
        if func["name"].replace(" ", "") == dict_for_map_many_func[target_name]:
            lines[idx][4 + sub_task_number] = func["name"].replace(" ", "")
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return True
        else:
            lines[idx][4 + sub_task_number] = func["name"].replace(" ", "")
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return False
