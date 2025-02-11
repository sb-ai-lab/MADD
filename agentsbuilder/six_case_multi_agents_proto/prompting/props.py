props_name = [
    "Docking score",
    "QED",
    "Synthetic Accessibility",
    "PAINS",
    "SureChEMBL",
    "Glaxo",
    "Diversity",
    "Brenk",
    "BBB",
    "KI",
    "IC50",
]
QED = """
QED 
QED is a quantitative metric that assesses how "drug-like" a compound is based on its physicochemical properties. 
A higher QED score suggests that a compound has properties favorable for development as an oral drug.
Range: [0; 1]
"""

IC50 = """
IC50 
The IC50 (half-maximal inhibitory concentration) value is a measure of the concentration of a drug or compound 
required to inhibit a particular biological or biochemical process by 50%
Value: 1/0
"""
docking_score = """
Docking
The scoring function used to predict the binding affinity of both ligand and target once it is docked
Range: (-inf, +inf)
Lower values correspond to high binding affinity.
"""

synthetic_accessibility = """
SA Score 
A score that estimates ease of synthesis (synthetic accessibility) of drug-like molecules
Range: [1; 10]
Molecules with the high SAscore (say, above 6) are difficult to synthesize, whereas, 
molecules with the low SAscore values are easily synthetically accessible.
"""

PAINS = """
PAINS
Pan-assay interference compounds (PAINS) are chemical compounds that often give false positive 
results in high-throughput screens.
Value: 1/0
1 - a molecule will probably give false-positive result in high-throughput screening
"""

SureChEMBL = """
SureChEMBL
SureChEMBL is a publicly available large-scale resource containing compounds extracted 
from the full text, images and attachments of patent documents.
Value: 1/0
1 - a molecule is under patent
"""

Glaxo = """
Glaxo (GlaxoSmithKline)
Glaxo filters are designed to exclude problem compounds, including classes known to be unstable
Value: 1/0
1 - a molecule is unstable
"""

BBB = """
BBB 
Blood–brain barrier (BBB) is a natural protective membrane that prevents the central nervous system 
(CNS) from toxins and pathogens in blood. 
Value: 1/0
1 - a molecule penetrates BBB (good for CNS drugs)
"""

Brenk = """
Brenk
The Brenk filter removes molecules containing substructures with undesirable pharmacokinetics or toxicity. 
Value: 1/0
1 - a molecule has substructures with undesirable pharmacokinetics or toxicity
"""

KI = """
Ki (Inhibition Constant)
Inhibitory constant (Ki) represents the concentration at which the inhibitor ligand occupies 
50 procentage of the receptor sites when no competing ligand is present.
Range: (0, inf)
Ki < 1 µm: High inhibition efficiency. 1 µm < Ki < 10 µm: Moderate efficiency. Ki > 10 microns: Low efficacy.
"""

props_descp_dict = {
    "Docking score": docking_score,
    "QED": QED,
    "Synthetic Accessibility": synthetic_accessibility,
    "PAINS": PAINS,
    "SureChEMBL": SureChEMBL,
    "Glaxo": Glaxo,
    "Brenk": Brenk,
    "BBB": BBB,
    "KI": KI,
    "IC50": IC50,
}


enter = """

"""
