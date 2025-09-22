import os
from typing import Dict, List, Optional
from urllib.parse import quote

import pandas as pd
import requests
from langchain.tools import tool

VALID_AFFINITY_TYPES = ["Ki", "Kd", "IC50"]


@tool
def fetch_BindingDB_data(params: Dict) -> str:
    """
    Tool for retrieving protein affinity data from BindingDB.

    This tool:
    1. Takes a protein name as input or a UniProt ID
    2. Queries UniProt to find the corresponding UniProt ID (if not provided)
    3. Retrieves specified affinity values (Ki, Kd, or IC50) for the protein from BindingDB
    4. Returns structured data about ligands and their affinity measurements

    Data source: BindingDB (https://www.bindingdb.org) - a public database of measured binding affinities

    Args:
        params: Dictionary containing:
            - protein_name: Name of the target protein (required)
            - affinity_type: Type of affinity measurement (Ki, Kd, or IC50, default: Ki)
            - cutoff: Optional affinity threshold in nM (default: 10000)
            - id: Optional, UniProt ID

    Returns:
        str: Succes or not
    """

    try:
        try:
            # parameter validation
            protein_name = params.get("protein_name")
            if not protein_name:
                print("Protein name not provided")
        except:
            pass

        affinity_type = params.get("affinity_type", "Ki")
        if affinity_type not in VALID_AFFINITY_TYPES:
            print(
                f"Invalid affinity type. Must be one of: {', '.join(VALID_AFFINITY_TYPES)}"
            )
            return False

        cutoff = params.get("cutoff", 10000)

        # Step 1: Get UniProt ID
        uniprot_id = params.get("id", False)
        if not uniprot_id:
            print("Starting search for ID of protein...")
            uniprot_id = fetch_uniprot_id(protein_name)
            if not uniprot_id:
                print(f"No UniProt ID found for {protein_name}")
                return False
            else:
                print("ID is: ", uniprot_id)

        # Step 2: Retrieve affinity data from BindingDB
        affinity_entries = fetch_affinity_bindingdb(uniprot_id, affinity_type, cutoff)
        pd.DataFrame(affinity_entries).to_csv(
            f'multi_agent_system/MADD_main/data_from_chem_db/molecules_{params.get("protein_name")}.csv'
        )

        txt_report = (
            f"Found {len(affinity_entries)} entrys for {protein_name}. Saved to "
            + f'multi_agent_system/MADD_main/data_from_chem_db/molecules_{params.get("protein_name")}.csv'
        )
        print(txt_report)

        os.environ["DS_FROM_BINDINGDB"] = (
            f'multi_agent_system/MADD_main/data_from_chem_db//molecules_{params.get("protein_name")}.csv'
        )
        return txt_report

    except Exception as e:
        return f"Processing error: {str(e)}"


def fetch_uniprot_id(protein_name: str) -> Optional[str]:
    """
    Get UniProt ID by UniProt REST API.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{protein_name} AND organism_id:9606",  # people
        "format": "json",
        "size": 1,
        "fields": "accession",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("results"):
            return data["results"][0].get("primaryAccession")
        return None

    except requests.exceptions.RequestException:
        return None


def fetch_affinity_bindingdb(
    uniprot_id: str, affinity_type: str, cutoff: int
) -> List[Dict]:
    """
    Retrieve affinity values from BindingDB for the given UniProt ID.

    Args:
        uniprot_id: UniProt accession ID
        affinity_type: Type of affinity measurement (Ki, Kd, or IC50)
        cutoff: Affinity threshold in nM

    Returns:
        List of dictionaries containing affinity data
    """
    url = f"http://bindingdb.org/rest/getLigandsByUniprots?uniprot={uniprot_id}&cutoff={cutoff}&response=application/json"

    try:
        response = requests.get(url, timeout=1200)
        response.raise_for_status()
        data = response.json()
        result = [
            i
            for i in data["getLindsByUniprotsResponse"]["affinities"]
            if i["affinity_type"] == affinity_type
        ]
        print(
            f"Found {len(result)} affinities for {uniprot_id} with type {affinity_type}"
        )
        return result

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 502:
            print("BindingDB server is temporarily unavailable (502 Bad Gateway)")
        else:
            print(f"HTTP error occurred: {e}")
        return []


@tool
def fetch_chembl_data(
    target_name: str, target_id: str = "", affinity_type: str = "Ki"
) -> str:
    """Get Ki for activity by current protein from ChemBL database. Return
    dict with smiles and Ki values, format: [{"smiles": smiles, affinity_type: affinity_valie, "affinity_units": affinity_units}, ...]

    Args:
        target_name: str, name of protein,
        target_id: optional, id of current protein from ChemBL. Don't make it up yourself!!! Only user can ask!!!
        affinity_type: optional, str, type of affinity measurement (default: 'Ki').
    """
    BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"

    if target_id == "" or target_id == None or target_id == False:
        # search target_id by protein name
        target_search = requests.get(
            f"{BASE_URL}/target/search?q={quote(target_name)}&format=json&limit=1000"
        )
        targets = target_search.json()["targets"]

        if not targets:
            print(f"Target '{target_name}' not found in ChEMBL")
            return []

        # get just first res
        target_id = targets[0]["target_chembl_id"]
        print(f"Found target: {targets[0]['pref_name']} ({target_id})")

    # get activity with Ki
    activities = []
    offset = 0
    while True:
        response = requests.get(
            f"{BASE_URL}/activity.json?"
            f"target_chembl_id={target_id}&"
            f"standard_type={affinity_type}&"
            f"offset={offset}&"
            "include=molecule"
        )

        data = response.json()
        activities += data["activities"]

        if not data["page_meta"]["next"]:
            break
        offset += len(data["activities"])

    # get SMILES and affinity values
    results = []
    for act in activities:
        try:
            smiles = act["canonical_smiles"]
            affinity_valie = act["standard_value"]
            affinity_units = act["standard_units"]
            results.append(
                {
                    "smiles": smiles,
                    affinity_type: affinity_valie,
                    "affinity_units": affinity_units,
                }
            )
        except (KeyError, TypeError):
            continue

    if len(results) < 1:
        return "No results found from ChemBL!"

    pd.DataFrame(results).to_csv(f"multi_agent_system/MADD_main/data_from_chem_db/molecules_{target_name}.csv")

    txt_report = (
        f"Found {len(results)} entrys for {target_name}. Saved to "
        + f"multi_agent_system/MADD_main/data_from_chem_db/molecules_{target_name}.csv"
    )
    print(txt_report)

    os.environ["DS_FROM_CHEMBL"] = f"multi_agent_system/MADD_main/data_from_chem_db/molecules_{target_name}.csv"
    return txt_report


if __name__ == "__main__":
    import os

    DATASET_DIR = "data_store/datasets"
    PROTEIN_NAME = "KRAS"
    AFFINITY_TYPE = "IC50"
    params = {
        "protein_name": PROTEIN_NAME,
        "affinity_type": AFFINITY_TYPE,
        "cutoff": 10000,
    }

    binding_data = fetch_BindingDB_data(params)
    print(f"Data fetched: {len(binding_data)} entries")

    # fetch_chembl_data('KRAS', affinity_type="IC50")

    # Save data to Excel
    df = pd.DataFrame(
        [
            {"Ligand": entry["ligand"], "Affinity": entry["affinity_value"]}
            for entry in binding_data
        ]
    )
    file_path = os.path.join(DATASET_DIR, f"sars_cov_2_ic50_data.xlsx")
    df.to_excel(file_path, index=False)
    print(f"Data saved to: {file_path}")
