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
            'Alzhmr' - generation of drug molecules for the treatment of Alzheimer's disease. \
            GSK-3beta inhibitors with high activity. \
            These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability.\
            Compounds that contain heterocyclic moieties to enhance binding affinity for amyloid-beta aggregates.\
            Hot keys: inhibitory activity against glycogen synthase kinase 3 beta (GSK-3β); compound should \
            demonstrate a permeability coefficient of at least 1.5 to ensure effective crossing of the blood-brain barrier; \
            tau protein kinases with an IC50 value lower than 50 nM.\
            'Sklrz' - Generation of molecules for the treatment of multiple sclerosis.\
            There are high activity tyrosine-protein kinase BTK inhibitors or highly potent non-covalent \
            BTK tyrosine kinase inhibitors from the TEC family of tyrosine kinases that have the potential \
            to affect B cells as a therapeutic target for the treatment of multiple sclerosis.\
            Hot keys: high activity tyrosine-protein kinase BTK inhibitors;  inhibit Bruton's tyrosine kinase \
            through non-covalent interaction; non-covalent BTK inhibitors with enhanced permeability across the blood-brain \
            barrier and high selectivity for Cytoplasmic tyrosine-protein kinase BMX;  immune signaling pathways \
            to treat multiple sclerosis.\
            'Prkns' - Generation of molecules for the treatment of Parkinson's disease.\
            These compounds should possess high bioavailability, cross the blood-brain barrier efficiently, and show \
            minimal metabolic degradation.\
            Hot keys: derivatives from amino acid groups with modifications in side chains to enhance bioactivity; \
            heterocyclic compounds featuring nitrogen and sulfur to improve pharmacokinetics; molecules using a \
            fragment-based approach, combining elements of natural alkaloids; molecules with properties of glutamate \
            receptor antagonists for neuroprotection; compounds that inhibit monoamine oxidase B (MAO-B);\
            dopamine agonist with improved blood-brain barrier penetration; dual-action molecule combining D2 \
            receptor agonism and A2A antagonism;  PDE10A inhibitor with enhanced selectivity and potency.\
            'Cnsr' - Generation of molecules for the treatment of lung cancer. \
            Molecules are inhibitors of KRAS protein with G12C mutation. \
            The molecules are selective, meaning they should not bind with HRAS and NRAS proteins.\
            Its target KRAS proteins with all possible mutations, including G12A/C/D/F/V/S, G13C/D, \
            V14I, L19F, Q22K, D33E, Q61H, K117N, G12C and A146V/T.\
            Hot keys: HRAS and NRAS proteins; KRAS G12C protein mutation, which drives cancer growth in lung cancer;\
            avoiding binding to HRAS and NRAS; low cross-reactivity with other RAS isoforms;  molecules to specifically bind and inhibit KRAS G12C\
            'Dslpdm' - Generation of molecules for the treatment of dyslipidemia.\
            Molecules that inhibit Proprotein Convertase Subtilisin/Kexin Type 9 with enhanced bioavailability and \
            the ability to cross the BBB. Molecules have affinity to the protein ATP citrate synthase, enhances reverse cholesterol transport via ABCA1 upregulation\
            , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.\
            Hot keys: molecules that disrupt the interaction between CD36 and oxidized LDL; ligands for SREBP-1c \
            inhibition to regulate fatty acid and triglyceride synthesis; dual-action agents that modulate both HDL \
            and LDL for improved cardiovascular outcomes; AMPK pathway to promote fatty acid oxidation;\
            IC50 value lower than 50 µM for inhibiting CETP;  Tmax of less than 2 hours for rapid action in lipid regulation;\
            a negative logD at pH 7.4 for improved selectivity in tissues; ANGPTL3 inhibitor to reduce plasma triglycerides;\
            PPARα agonist with reduced side;  ApoC-III antisense oligonucleotide with enhanced cellular uptake.\
            'TBLET' - Generation of molecules for acquired drug resistance. \
            Molecules that selectively induce apoptosis in drug-resistant tumor cells.\
            It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.\
            Hot keys: molecular structures targeting drug resistance mechanisms in cancer cells;\
            ABC transporters involved in multidrug resistance; molecules that selectively induce apoptosis in drug-resistant tumor cells;\
            counteracting drug resistance in oncological therapies; treatment sensitivity in resistant tumors; compounds that enhance the \
            efficacy of existing anti-resistance treatments; synergistic compound that significantly enhances the activity of existing therapeutic \
            agents against drug-resistant pathogens; selectively target the Ras-Raf-MEK-ERK signaling pathway\
            molecules targeting the efflux pumps responsible for drug resistance.\
            'RNDM' - generation with random properties",
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
    "description": "Generation of drug molecules for the treatment of Alzheimer's disease. \
        GSK-3beta inhibitors with high activity. \
        These molecules can bind to GSK-3beta protein, molecules has low brain-blood barrier permeability.\
        Compounds that contain heterocyclic moieties to enhance binding affinity for amyloid-beta aggregates.\
        Hot keys: inhibitory activity against glycogen synthase kinase 3 beta (GSK-3β); compound should \
        demonstrate a permeability coefficient of at least 1.5 to ensure effective crossing of the blood-brain barrier; \
        tau protein kinases with an IC50 value lower than 50 nM.",
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
    "description": "Generation of molecules for the treatment of Parkinson's disease.\
        These compounds should possess high bioavailability, cross the blood-brain barrier efficiently, and show \
        minimal metabolic degradation.\
        Hot keys: derivatives from amino acid groups with modifications in side chains to enhance bioactivity; \
        heterocyclic compounds featuring nitrogen and sulfur to improve pharmacokinetics; molecules using a \
        fragment-based approach, combining elements of natural alkaloids; molecules with properties of glutamate \
        receptor antagonists for neuroprotection; compounds that inhibit monoamine oxidase B (MAO-B);\
        dopamine agonist with improved blood-brain barrier penetration; dual-action molecule combining D2 \
        receptor agonism and A2A antagonism;  PDE10A inhibitor with enhanced selectivity and potency.",
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
            to affect B cells as a therapeutic target for the treatment of multiple sclerosis.\
            Hot keys: high activity tyrosine-protein kinase BTK inhibitors;  inhibit Bruton's tyrosine kinase \
            through non-covalent interaction; non-covalent BTK inhibitors with enhanced permeability across the blood-brain \
            barrier and high selectivity for Cytoplasmic tyrosine-protein kinase BMX;  immune signaling pathways \
            to treat multiple sclerosis.",
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
            , inhibits HMG-CoA reductase with improved safety profile compared to statins. It can be  PCSK9 inhibitors to enhance LDL receptor recycling and reduce LDL cholesterol levels.\
            Hot keys: molecules that disrupt the interaction between CD36 and oxidized LDL; ligands for SREBP-1c \
            inhibition to regulate fatty acid and triglyceride synthesis; dual-action agents that modulate both HDL \
            and LDL for improved cardiovascular outcomes; AMPK pathway to promote fatty acid oxidation;\
            IC50 value lower than 50 µM for inhibiting CETP;  Tmax of less than 2 hours for rapid action in lipid regulation;\
            a negative logD at pH 7.4 for improved selectivity in tissues; ANGPTL3 inhibitor to reduce plasma triglycerides;\
            PPARα agonist with reduced side;  ApoC-III antisense oligonucleotide with enhanced cellular uptake; ",
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
        It significantly enhances the activity of existing therapeutic agents against drug-resistant pathogens.\
        Hot keys: molecular structures targeting drug resistance mechanisms in cancer cells;\
        ABC transporters involved in multidrug resistance; molecules that selectively induce apoptosis in drug-resistant tumor cells;\
        counteracting drug resistance in oncological therapies; treatment sensitivity in resistant tumors; compounds that enhance the \
        efficacy of existing anti-resistance treatments; synergistic compound that significantly enhances the activity of existing therapeutic \
        agents against drug-resistant pathogens; selectively target the Ras-Raf-MEK-ERK signaling pathway\
            molecules targeting the efflux pumps responsible for drug resistance.\
            ",
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
            V14I, L19F, Q22K, D33E, Q61H, K117N, G12C and A146V/T.\
            Hot keys: HRAS and NRAS proteins; KRAS G12C protein mutation, which drives cancer growth in lung cancer;\
            avoiding binding to HRAS and NRAS; low cross-reactivity with other RAS isoforms;  molecules to specifically bind and inhibit KRAS G12C\
            ",
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
