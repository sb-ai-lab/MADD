mols_gen_metadata = {
    "name": "request_mols_generation",
    "description": "Generates molecules. It need when users ask to give a molecule without indicating why and what kind of molecule it is",
    "arguments": [
        {"name": "num", "type": "int", "description": "number of molecules to generate"}
    ],
}
mols_gen_total_metadata = {
    "name": "gen_mols_all_case",
    "description": "Determines the properties of molecules. Use when user want to define propeties. it is forbidden to call if there are no molecules",
    "arguments": [
        {
            "name": "generation_type",
            "case": "str",
            "description": "Could be one of: 'Alzhmr','Sklrz','Prkns','Cnsr','Dslpdm','TBLET', 'RNDM'.\
            'Alzhmr' - generation of drug molecules for the treatment of Alzheimer's disease. GSK-3beta inhibitors with high activity. \
            These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability.\
            'Sklrz' - generation of molecules for the treatment of multiple sclerosis.\
            There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
            BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
            to affect B cells as a therapeutic target for the treatment of multiple sclerosis.\
            'Prkns' - generation of molecules for the treatment of Parkinson's disease.\
            'Cnsr' - generation of molecules for the treatment of lung cancer. \
            Molecules are inhibitors of KRAS protein with G12C mutation. \
            The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
            Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
            V14I, L19F, Q22K, D33E, Q61H, K117N and A146V/T.\
            'Dslpdm' - Generation of molecules for the treatment of dyslipidemia.\
            Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
            the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
            , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.\
            'TBLET' - generation of molecules for acquired drug resistance. Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
            It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.\
            'RNDM' - random generation.",
        },
        {
            "name": "num",
            "type": "int",
            "description": "Number of molecules for generation",
        },
    ],
}
request_check_props_metadata = {
    "name": "request_check_props",
    "description": "Determines the properties of molecules. Use when user want to define propeties. it is forbidden to call if there are no molecules",
    "arguments": [
        {
            "name": "molecules",
            "type": "str",
            "description": "Molecules in SMILES format separated by comma. You may not use ellipses or abbreviations.",
        }
    ],
}
gen_mols_alzheimer_metadata = {
    "name": "gen_mols_alzheimer",
    "description": "Generation of drug molecules for the treatment of Alzheimer's disease. GSK-3beta inhibitors with high activity. \
            These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability",
    "arguments": [
        {
            "name": "num",
            "type": "int",
            "description": "number of molecules to generate",
        }
    ],
}
gen_mols_parkinson_metadata = {
    "name": "gen_mols_parkinson",
    "description": "Generation of molecules for the treatment of Parkinson's disease.",
    "arguments": [
        {
            "name": "num",
            "type": "int",
            "description": "number of molecules to generate",
        }
    ],
}
gen_mols_multiple_sclerosis_metadata = {
    "name": "gen_mols_multiple_sclerosis",
    "description": "Generation of molecules for the treatment of multiple sclerosis.\
            There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
            BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
            to affect B cells as a therapeutic target for the treatment of multiple sclerosis.",
    "arguments": [
        {
            "name": "num",
            "type": "int",
            "description": "number of molecules to generate",
        }
    ],
}
gen_mols_dyslipidemia_metadata = {
    "name": "gen_mols_dyslipidemia",
    "description": "Generation of molecules for the treatment of dyslipidemia.\
            Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
            the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
            , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.",
    "arguments": [
        {
            "name": "num",
            "type": "int",
            "description": "number of molecules to generate",
        }
    ],
}
gen_mols_acquired_drug_resistance_metadata = {
    "name": "gen_mols_acquired_drug_resistance",
    "description": "Generation of molecules for acquired drug resistance. \
        Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
        It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.",
    "arguments": [
        {
            "name": "num",
            "type": "int",
            "description": "number of molecules to generate",
        }
    ],
}
gen_mols_lung_cancer_metadata = {
    "name": "gen_mols_lung_cancer",
    "description": "Generation of molecules for the treatment of lung cancer. \
            Molecules are inhibitors of KRAS protein with G12C mutation. \
            The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
            Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
            V14I, L19F, Q22K, D33E, Q61H, K117N and A146V/T.",
    "arguments": [
        {
            "name": "num",
            "type": "int",
            "description": "number of molecules to generate",
        }
    ],
}
make_answer_chat_model_metadata = {
    "name": "make_answer_chat_model",
    "description": "Answers a question using a chat model (chemical assistant)). \
    Suitable for free questions that do not require calling other tools or insufficient/incorrect arguments to call the function. \
    Use it if you need to tell a person how to use what function (if there is doubt in his question)! \
    Use this function if the user asks to define properties for molecules but does not provide them in Smiles format. \
    For example, if user ask: 'What needs to be done for you to define the properties?';\n \
    'What is needed to generate molecules?;'",
    "arguments": [
        {
            "name": "msg",
            "type": "str",
            "description": "Message from user",
        }
    ],
}
