gen_mols_metadata = {
    "name": "gen_mols",
    "description": "Generate molecules by generative models. Inference.",
    "arguments": [
        {
            "name": "case",
            "case": "str",
            "description": "Could be one of: 'Alzhmr','Sklrz','Prkns','Cnsr','Dslpdm','TBLET', 'RNDM', 'ANOTHER'.\
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
            'ANOTHER' - (Can be any word (disease that the user mentioned user).\
            'RNDM' - random generation .",
        },
        {
            "name": "num",
            "type": "int",
            "description": "Number of molecules for generation",
        },
        {
            "name": "model",
            "type": "str",
            "description": "Model for generation, can be: \
            'CVAE' - ideal for Dyslipidemia, Lung Cancer case, Drug resistance;\
            this method shows the following metrics depending on the choice of case (disease):\
            Filter group	1	2	3	4	5	Diversity\
            Alzheimer's	26,06%	23,58%	18,47%	18,15%	18,15%	\
            Sclerosis 	15,43%	13,75%	13,32%	13,29%	13,29%	0,25\
            Lung cancer	6,12%	5,72%	4,97%	4,76%	4,76%	0,8\
            Drug resistance	8,32%	6,92%	6,14%	6,05%	6,05%	0,77\
            Dyslipidemia	28,87%	28,27%	25,07%	24,50%	13,16%	0,205\
            Parkinson	3,32%	3,06%	2,69%	2,65%	2,65%	0,235;\
            'LSTM' - this method shows the following metrics depending on the choice of case (disease):\
            Filter group	1	2	3	4	5	Diversity\
            Alzheimer's	19,03%	14,75%	11,70%	11,32%	11,32%	0,37\
            Sclerosis 	5,90%	4,35%	3,49%	3,36%	3,36%	0,39\
            Lung cancer	5,53%	4,41%	3,43%	3,31%	3,31%	0,39\
            Drug resistance	0,23%	0,15%	0,10%	0,10%	0,10%	0,39\
            Dyslipidemia	7,27%	6,15%	4,92%	4,72%	4,72%	0,34\
            Parkinson	14,45%	11,48%	8,92%	8,57%	8,57%	0,36;\
            'RL' - ideal for Sclerosis case: \
            Filter group	1	2	3	4	5	Diversity\
            Alzheimer's	15,8	14,34	10,99	10,74	10,74	0,21\
            Sclerosis	22.81	20.34	18.39	18.22	18.22	0.11\
            Lung cancer	0,57	0,53	0,51	0,51%	0,51%	0,088\
            Drug resistance	0,63	0,52%	0,4	0,38	0,38	0,13\
            Dyslipidemia	0,02	0,02	0,02	0,02	0,02	0,045\
            Parkinson	0.03	0.03	0	0	0	0.17  ; \
            'GraphGA' - ideal for Alzheimer's case:\
            Filter group	1	2	3	4	5	Diversity\
            Alzheimer's	69,00%	69,00%	37,00%	28,00%	28,00%	0,18\
            Sclerosis 	0,00%	0,00%	0,00%	0,00%	0,00%	0,14\
            Lung cancer	1,00%	1,00%	1,00%	1,00%	1,00%	0,11\
            Drug resistance	0,00%	0,00%	0,00%	0,00%	0,00%	0,11\
            Dyslipidemia	0,00%	0,00%	0,00%	0,00%	0,00%	0,063\
            Parkinson	24,00%	24,00%	12,00%	12,00%	12,00%	0,16.",
        },
    ],
}
train_gen_models_metadata = {
    "name": "train_gen_models",
    "description": "Train a generative model with a custom dataset (only if the user requests training).",
    "arguments": [
        {
            "name": "model",
            "type": "str",
            "description": "Model for finetuning for specific not existed case. Available: 'RL', 'CVAE', 'LSTM', 'GraphGA'",
        },
        {
            "name": "epoch",
            "type": "int",
            "description": "Number of train epoch",
        },
        {
            "name": "case_name",
            "type": "str",
            "description": "The name of the disease for which the model will be trained (in the future, the user will ask for inference using this name)",
        }
    ],
}

automl_predictive_models_metadata = {
    "name": "automl_predictive_models",
    "description": "Calling the AUTOML module to automatically train a predictive model for a property that is not represented in the current tools.",
    "arguments": [
        {
            "name": "property",
            "type": "str",
            "description": "Name of property for prediction. Can be any property name based on user query.",
        }
    ],
}

inference_predictive_models_metadata = {
    "name": "inference_predictive_models",
    "description": "Predicting properties using ready-made ML models. Only suitable for IC50 property.",
    "arguments": [
        {
            "name": "property",
            "type": "str",
            "description": "Name of property for prediction. Available: 'IC50'",
        },
                {
            "name": "molecules",
            "type": "list",
            "description": "List of molecules in SMILES format",
        }
    ],
}

compute_docking_score_metadata = {
    "name": "compute_docking_score",
    "description": "Calculation of docking score for molecules",
    "arguments": [
        {
            "name": "molecules",
            "type": "str",
            "description": "List of molecules in SMILES format",
        }
    ],
}

compute_by_rdkit_metadata = {
    "name": "compute_by_rdkit",
    "description": "Prediction of properties for molecules. The following are available: 'Brenk', 'Diversity', 'PAINS', 'SureChEMBL', 'Glaxo', 'SA', 'QED'.",
    "arguments": [
        {
            "name": "molecules",
            "type": "str",
            "description": "List of molecules in SMILES format.",
        },
        {
            "name": "property",
            "type": "str",
            "description": "Name of property for prediction. Available: 'Brenk', 'Diversity', 'PAINS', 'SureChEMBL', 'Glaxo', 'SA', 'QED'.",
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
