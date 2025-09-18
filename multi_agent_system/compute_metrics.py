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


def add_answers(answers: list, path: str):
    llm_res, tool_res, check = answers
    pd.DataFrame(
        {"llm_result": llm_res, "tool_result": tool_res, "check": check}
    ).to_excel(path)


def validate_decompose(
    idx: int,
    decompose_lst: list,
    validation_path="multi_agents_system/testcase/experiment3.xlsx",
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
    path_total_val="./experiment3.xlsx",
    func_field: str = "action",
    parm_field: str = "action_input",
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
    if func[func_field].replace(" ", "") == "make_answer_chat_model":
        lines[idx][4 + sub_task_number] = func[func_field].replace(" ", "")
        pd.DataFrame(
            lines,
            columns=columns,
        ).to_excel(path_total_val, index=False)
        return False
    # if few function with a lot of args
    elif func[func_field].replace(" ", "") == "gen_mols_all_case":
        if func[parm_field]["generation_type"] == dict_for_map_few_func[target_name]:
            lines[idx][4 + sub_task_number] = func[parm_field]["generation_type"]
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return True
        else:
            lines[idx][4 + sub_task_number] = func[parm_field]["generation_type"]
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return False
    # if many function with a few args
    else:
        if func[func_field].replace(" ", "") == dict_for_map_many_func[target_name]:
            lines[idx][4 + sub_task_number] = func[func_field].replace(" ", "")
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return True
        else:
            lines[idx][4 + sub_task_number] = func[func_field].replace(" ", "")
            pd.DataFrame(
                lines,
                columns=columns,
            ).to_excel(path_total_val, index=False)
            return False


def compute_metrics(
    model_name: str = "no_name_model",
    file_path: str = "multi_agents_system/testcase/experiment3_example.xlsx",
    just_one_task_per_q: bool = False,
):
    """
    Compute pipeline metrics

    Parameters
    ----------

    file_path : str
        Path to excel file with next columns:
        case, content, decompose_tasks, is_correct, task 1, task 2, task 3, task 4
    model_name : str
        Name of model with which testing was carried out
    just_one_task_per_q: bool, optional
        True, if passed querys just with 1 task
    """
    just_1_case_in_all_smpls = True
    dfrm = pd.read_excel(file_path)

    number_subtasks = 0
    number_tasks = 0

    correct_subtasks = 0
    correct_tasks = 0

    decomposer_true = 0

    # add zeros columns for result
    dfrm["conductors_score"] = 0
    dfrm["score_from"] = 0
    dfrm["total_score"] = 0
    columns = dfrm.columns

    lst = dfrm.values.tolist()

    for row in lst:
        try:
            funcs = []
            cases = (
                row[0]
                .replace("Parkinson ", "Parkinson")
                .replace("Drug_Resistance ", "Drug_Resistance")
                .split(", ")
            )
            if row[3] == 1.0:
                decomposer_true += row[3]

            row[11 - 1] = len(cases)
            [funcs.append(i) for i in row[4:9] if isinstance(i, str)]

            # for every subtask in main task(query)
            for n, case in enumerate(cases):
                if just_one_task_per_q:
                    is_correct = dict_for_map_many_func[case] == row[4 + n] and len(
                        funcs
                    ) == len(cases)
                else:
                    is_correct = dict_for_map_many_func[case] == row[4 + n]
                row[10 - 1], correct_subtasks = (
                    row[10 - 1] + int(is_correct),
                    correct_subtasks + int(is_correct),
                )

            # if all subtasks are defined correctly
            if row[10 - 1] == row[11 - 1] and len(funcs) == len(cases):
                correct_tasks += 1
                row[12 - 1] = 1
            else:
                row[12 - 1] = 0

            if just_1_case_in_all_smpls:
                if len(cases) > 1:
                    just_1_case_in_all_smpls = False

            number_subtasks, number_tasks = (
                number_subtasks + len(cases),
                number_tasks + 1,
            )

        except:
            continue

    pd.DataFrame(lst, columns=columns).to_excel(
        f"./result_{model_name}.xlsx", index=False
    )

    if not (just_1_case_in_all_smpls):
        print(
            "Percentage true subtasks by Conductor: ",
            100 / (number_subtasks) * correct_subtasks,
        )
        print(
            "Percentage true tasks by Decomposer: ",
            100 / number_tasks * decomposer_true,
        )
        print(
            "Percentage true tasks by Conductor: ",
            100 / (number_tasks) * correct_tasks,
        )
        return 100 / (number_tasks) * correct_tasks
    else:
        print("Percentage true tasks: ", 100 / number_tasks * correct_tasks)


# for validation Summarization
def check_total_answer(true_answers: list[str], total_answer: str) -> bool:
    for true_ans in true_answers:
        if not (true_ans in total_answer):
            return False
    return True


def exctrac_mols_and_props(mols: str) -> list:
    pattern = r"\| ([A-Za-z0-9@+\-=#\[\]\(\)\\\/\.\*]+) \|"

    molecules = re.findall(pattern, mols)
    valid_molecules = [mol for mol in molecules if any(c.isalpha() for c in mol)]
    return valid_molecules


if __name__ == "__main__":
    cond_per_tasks = compute_metrics("llama", "/home/alina/Desktop/LLMagentsBuilder/experiment2_2.xlsx")
    res = pd.read_excel("/home/alina/Desktop/LLMagentsBuilder/experiment2_2.xlsx").values.tolist()
    
    total_succ = 0
    success_by_tasks = 0
    empty_sample = 0
    # cnt for count tasks
    cnt = 0

    for ex in cond_per_tasks:
        succ = 0
        for ans_llm, func_ans in zip([ex[1]], [ex[2]]):
            is_correct = check_total_answer(exctrac_mols_and_props(func_ans), ans_llm)

            succ += is_correct
            success_by_tasks += is_correct

            cnt += 1

        if eval(ex[1]) == []:
            empty_sample += 1
        elif succ == len(eval(ex[1])):
            total_succ += 1

    print("Summ correct by tasks: ", 1 / cnt * success_by_tasks * 100 )
    print("Total accuracy by tasks: ", cond_per_tasks * 1 / cnt * success_by_tasks)
