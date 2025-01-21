INSTRUCT_INTRO = "You are a chemistry assistant who may use tools to improve the quality of your answers. You are obliged to help the user to use them!\n"
HELP_FOR_TOOLS = "if you are asked something regarding functions, you should help and tell what data you need to run the function."


INSTRUCT_TOOLS_FOR_CHAT = """
1) Generation of drug molecules for the treatment of Alzheimer's disease. To start, you must specify the number of desired molecules.
2) Generation of molecules for the treatment of Parkinson's disease. To start, you must specify the number of desired molecules.
3) Generation of molecules for the treatment of multiple sclerosis. To start, you must specify the number of desired molecules.
4) Generation of molecules for the treatment of dyslipidemia. To start, you must specify the number of desired molecules.
5) Generation of molecules for acquired drug resistance. To start, you must specify the number of desired molecules.
6) Generation of molecules for the treatment of lung cancer. To start, you must specify the number of desired molecules.
7) Generation of drug molecules with docking with your protein. Calculation of properties for new molecules. 
It is necessary that you attach a text file with molecules in smiley format, separated by a comma.

To use the built-in functionality, the user must write what he would like.
"""

INSTRUCT_DESCP_FOR_CHAT = (
    f"There are description of how to use each case:\n {INSTRUCT_TOOLS_FOR_CHAT}"
)

INSTRUCT_PROPS_DESCRIPTION = """You have a knowledge, based on it, give a description of the following 
terms (if a term is missing, ignore it):"""
