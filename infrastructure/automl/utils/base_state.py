import json
from typing import List, Union 
from pydantic import BaseModel
import os
from infrastructure.automl.utils.calculateble_prop_funcs import config
from huggingface_hub import HfApi

class BaseState(BaseModel):
    """
    Base class for representing a state in a machine learning pipeline.
    
        This class provides a basic structure for storing and managing the state of a 
        machine learning process, including model weights, architecture type, case name, and a status indicator.
    """

    status:str = None
    weights_path:str = {"regression" : None, "classification" : None} 
    arch_type:str = None
    case_name: str = None
    base_state:dict = {"description" : None,
                       "generative_models": {"data_path":None,"feature_column":None,"target_column":None,"problem":None,'status':status,'weights_path':None,'arch_type':arch_type,"metric":None},
                        "ml_models":{"data_path":None,"feature_column":None,"target_column":None,'status':status,'weights_path' : weights_path,"metric":None,
                            "Predictable properties" : {}
                                  }
                        }
class TrainState:
    """
    A class representing the current status of available models. It manages the loading, saving, and tracking of trained models, enabling efficient access to their configurations and capabilities. This class facilitates the organization and utilization of different model versions during experimentation and deployment.
    
        Returns:
            _type_: _description_
    """

    defult_parameters = BaseState()

    def __init__(self,state_path:str=None):
        """
        Initializes the AutoMLStateManager.
        
        Loads the state from a specified path or a default location if it exists,
        otherwise initializes a new state and saves it to the default location.
        This ensures the system can resume progress and maintain context across sessions.
        
        Args:
            state_path (str, optional): The path to the state file. If None, it attempts to load from
                "automl/state/state.json". Defaults to None.
        
        Returns:
            None
        """
        self.state_path = state_path
        if state_path is not None:
            self.current_state = self.__load_state()
        elif os.path.isfile("infrastructure/automl/state.json"):
            self.state_path = "infrastructure/automl/state.json"
            self.current_state = self.__load_state()
        else:
            self.state_path = r'infrastructure/automl/state.json'
            self.current_state = {"Calculateble properties" : config}
            self.__save_state()

    def __call__(self, case:str = None, 
                 model:str = None,
                   *args,
                     **kwargs):
        """
        Return the model's state, optionally filtered by task and model type.
        
        Args:
            case (str, optional): The name of the task to retrieve state for. If None, returns the full state. Defaults to None.
            model (str, optional):  Specifies whether to return states related to 'ml' (machine learning) or 'gen' (generative) models. 
                                    Must be provided along with `case`. Defaults to 'ml'.
        
        Returns:
            dict: The requested state information.  Can be the full state, a specific task's state, or a specific model's state within a task.
        """
        if case is None:
            return self.current_state
        else:
            if model=='ml':
                return self.current_state[case]["ml_models"]
            elif model=='gen':
                return self.current_state[case]["generative_models"]
            else: 
                return self.current_state[case]
    
    def add_new_case(self,
                     case_name:str,
                     rewrite=False,
                     description : str = 'Unknown case',
                     **kwargs):
        """
        Adds a new experimental setup to the tracked state, allowing for the storage and recall of different configurations.
        
        Args:
            case_name (str): A unique identifier for the experimental setup.
            rewrite (bool, optional): If True, overwrites an existing setup with the same name. Defaults to False.
            description (str, optional): A brief description of the experimental setup. Defaults to 'Unknown case'.
            **kwargs: Additional parameters that can be used to customize the setup.
        
        Returns:
            None:
        """
        if case_name in self.current_state.keys() and not rewrite:
            print(f'Case already exist! Change name for new case, or use exist case state named {case_name}!')
            print(f"Now using case named '{case_name}' - Case Description: {self.current_state[case_name]['description']}")
            return None
        
        self.current_state[case_name] = self.defult_parameters.base_state
        self.current_state[case_name]['description'] = description
        self.__save_state()

    def gen_model_upd_data(self,
                           case:str,
                           data_path:str=None,
                           feature_column:List[str] = None,
                           target_column:List[str] = None
                           ):
        """
        Updates the configuration for a generative model associated with a specific case.
        
          This method allows you to define or modify the data source, features, and target variables
          used for training or evaluating a generative model within the system.  It selectively updates
          the model's parameters based on the provided arguments.
        
        Args:
            case (str): The name or identifier of the case (generative model).
            data_path (str, optional): The file path to the training dataset. Defaults to None.
            feature_column (list[str], optional): A list of column names representing the input features. Defaults to None.
            target_column (list[str], optional): A list of column names representing the target variables. Defaults to None.
        
        Returns:
            None
        """
        if data_path is not None:
            self.current_state[case]["generative_models"]["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["generative_models"]["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["generative_models"]["target_column"] = target_column
        self.__save_state()

    def ml_model_upd_data(self,
                        case:str,
                        data_path:str=None,
                        feature_column:List[str] = None,
                        target_column:List[str] = None,
                        predictable_properties:dict = None):
        """
        Updates the configuration for machine learning models associated with a specific case.
        
        This method allows for flexible updates to the model training parameters, including the data source, 
        feature and target columns, and a set of properties the model should predict. 
        Updates can be partial, modifying only the parameters provided in the function call.
        
        Args:
            case (str): The name of the case to update the ML model parameters for.
            data_path (str, optional): The path to the training data file. Defaults to None.
            feature_column (list[str], optional): A list of column names to use as features for training. Defaults to None.
            target_column (list[str], optional): A list of column names to use as target variables for training. Defaults to None.
            predictable_properties (dict, optional): A dictionary defining the properties predictable for different problem types 
                (e.g., {"regression": ["LogP", "QED"], "classification": ["IC50", "Num rings"]}). Defaults to None.
        
        Returns:
            None
        """
        if data_path is not None:
            self.current_state[case]["ml_models"]["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["ml_models"]["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["ml_models"]["target_column"] = target_column
        if predictable_properties is not None:
            self.current_state[case]["ml_models"]['Predictable properties'] = predictable_properties
        self.__validate_properties(case)
        print(f"Data for ML models training has been updated! \
               \n Current predictable properties and tasks are {self.current_state[case]['ml_models']['Predictable properties']}")
        self.__save_state()
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.upload_file(
            path_or_fileobj="infrastructure/automl/state.json",
            repo_id="SoloWayG/Molecule_transformer",
            repo_type="model",
            path_in_repo = 'state.json'
        )

    def ml_model_upd_status(self,
                            case:str,
                            model_weight_path:str = None,
                            metric = None,
                            status:int = None,
                            problem:str = 'regression'):
        """
        Updates the training status and associated details of a machine learning model for a specific task.
        
        This method allows tracking the training progress of a model, recording the location of saved weights, and storing performance metrics. It ensures that the training status is updated only if the model hasn't already been fully trained. This provides a centralized way to manage and monitor the state of models used within the system.
        
        Args:
            case (str): The name of the task or experiment the model is associated with.
            model_weight_path (str, optional): The directory where the trained model weights are saved. Defaults to None.
            metric (float, optional): The performance metric achieved after training (e.g., R-squared, RMSE). Defaults to None.
            status (int, optional): An integer representing the training status: 0 for "Not Trained", 1 for "Training", and 2 for "Trained". Defaults to None.
            problem (str, optional): Type of the problem. Defaults to 'regression'.
        
        Returns:
            None
        """
        valid_status = ['Not Trained','Training','Trained']
        if status is not None and self.current_state[case]["ml_models"]['status'] != 'Trained':
            self.current_state[case]["ml_models"]['status'] = valid_status[status]
        elif self.current_state[case]["ml_models"]['status'] == 'Trained':
            print(f'ML model for task "{case}" already trained!')
        if model_weight_path is not None:
            if not os.path.isdir(model_weight_path ):
                    os.mkdir(model_weight_path )
        if self.current_state[case]["ml_models"]['weights_path'][problem] is None and model_weight_path is not None:
            self.current_state[case]["ml_models"]['weights_path'][problem] = model_weight_path 
        if not metric is None:
           self.current_state[case]["ml_models"]['metric'] =  metric
        self.__save_state()
        api = HfApi(token=os.getenv("HF_TOKEN"))
        api.upload_file(
            path_or_fileobj="infrastructure/automl/state.json",
            repo_id="SoloWayG/Molecule_transformer",
            repo_type="model",
            path_in_repo = 'state.json'
        )
            

    def gen_model_upd_status(self,
                             case:str,
                             model_weight_path:str = None,
                             metric = None):
        """
        Updates the training status of the generative model for a given case.
        
        This method tracks the progress of the generative model, transitioning its status from 'Training' to 'Trained' upon each call. It also allows storing the path to the model weights and the corresponding metric achieved after training, enabling a record of model performance.
        
        Args:
            case (str): The name of the case or task for which the generative model is being updated.
            model_weight_path (str, optional): The directory path where the trained model weights are saved.  If the directory doesn't exist, it will be created. Defaults to None.
            metric (float, optional): The value of the metric achieved after training the model. Defaults to None.
        
        Returns:
            None
        """

        if self.current_state[case]["generative_models"]['status'] is None:
            self.current_state[case]["generative_models"]['status'] = 'Training'
        elif self.current_state[case]["generative_models"]['status'] == 'Training':
            self.current_state[case]["generative_models"]['status'] = 'Trained'
        else:
            print(f'Generative model for task "{case}" already trained!')
        if model_weight_path is not None:
            if not os.path.isdir(model_weight_path):
                os.mkdir(model_weight_path )
        if not self.current_state[case]["generative_models"]['weights_path'] is None and model_weight_path is not None:
            self.current_state[case]["generative_models"]['weights_path'] = model_weight_path
        if not metric is None:
           self.current_state[case]["generative_models"]['metric'] =  metric
        self.__save_state()
    
    def show_calculateble_propreties(self):
        """
        Returns the keys of the calculable properties stored in the current state.
        
        Args:
            self: The instance of the TrainState class.
        
        Returns:
            list: A list of keys representing the names of the properties that can be calculated.
        """
        return self.current_state["Calculateble properties"].keys()

    @staticmethod
    def load_state(path:str = r'automl/state.json'):
        """
        Loads the state from a JSON file and augments it with project configuration data.
        
        Args:
            path (str, optional): The path to the JSON file containing the state. Defaults to 'automl/state.json'.
        
        Returns:
            dict: The loaded state dictionary, with an added "Calculateble properties" field containing the project configuration.
        """
        state = json.load(open(path))
        state["Calculateble properties"] = config
        return state
    

    def save(self,path:str = r'automl/state.json'):
        """
        Saves the current state of the object to a JSON file, excluding properties that are dynamically calculated.
        
        Args:
            path: The path to the JSON file where the state will be saved.
                Defaults to 'automl/state.json'.
        
        Returns:
            None
        """

        saving_dict = self.current_state.copy()
        del saving_dict["Calculateble properties"]
        json.dump(self.current_state, open(saving_dict, 'w' ) )

    def __save_state(self):
        """
        Saves the current state of the object to a JSON file.
        
        This method persists the object's configuration by serializing its core state 
        to a JSON file. It ensures that sensitive or derived data isn't saved, 
        focusing on the essential parameters needed for resuming or replicating the process.
        
        Args:
            self: The object instance.
        
        Returns:
            None
        """
        saving_dict = self.current_state.copy()
        del saving_dict["Calculateble properties"]
        json.dump(saving_dict, open(self.state_path, 'w' ) )

    def __load_state(self):
        """
        Loads a saved state from a JSON file and combines it with current settings.
        
        Args:
            self: The instance of the class.
        
        Initializes:
            self.state_path: Path to the JSON file containing the state.
        
        Returns:
            dict: The loaded state, updated with the current configuration.
        
        The method retrieves previously saved training parameters from a file.  This allows for resuming training from a specific point, avoiding redundant calculations and ensuring consistency across sessions. The loaded state is then augmented with the current configuration to ensure the training process utilizes the most up-to-date settings.
        """
        state = json.load(open(self.state_path))
        state["Calculateble properties"] = config
        return state
    
    def __validate_properties(self,case:str):
        """
        The function refines the list of properties to be predicted by removing those that can be reliably calculated from existing data. This ensures the machine learning models focus on learning relationships for properties not already determinable through established methods.
        
        Args:
            case (str): The name of the current case or experiment being processed.
        
        Returns:
            None: This method modifies the `current_state` attribute of the `TrainState` object in place, updating the 'Predictable properties' dictionary by removing calculated properties.
        """
        if self.current_state[case]["ml_models"]["feature_column"] is not None and self.current_state[case]["ml_models"]['Predictable properties'] != {}:
            temp_predictable_properties = {}
            for task in self.current_state[case]["ml_models"]['Predictable properties'].keys():
                temp_predictable_properties[task] = [proper for proper in self.current_state[case]["ml_models"]['Predictable properties'][task] if proper not in self.show_calculateble_propreties()]
                if len(temp_predictable_properties[task])==0:
                    del temp_predictable_properties[task]
            self.current_state[case]["ml_models"]['Predictable properties'] = temp_predictable_properties

if __name__=='__main__':
    task = 'Brain_cancer_test'
    feature_column=['canonical_smiles']
    target_column=['docking_score','canonical_smiles','QED','Synthetic Accessibility','PAINS','SureChEMBL','Glaxo','Brenk','IC50']
    regression_props = ['LogP','docking_score',"Synthetic Accessibility",'QED']
    classification_props = ['QED','QED']

    state = TrainState()
    state.add_new_case(task,rewrite=True)
    state.ml_model_upd_data(case=task,
                            feature_column=feature_column,
                            target_column=target_column,
                            predictable_properties={"regression":regression_props, "classification":classification_props})
    print(state(task,'ml')['Predictable properties'])
    state.gen_model_upd_data(case=task,data_path=r'D:\Projects\ChemCoScientist\Project\ChemCoScientist\automl\data\data_4j1r.csv')
    state.gen_model_upd_status(case=task)
    state.gen_model_upd_status(case=task)
    state.gen_model_upd_status(case=task)
    state.ml_model_upd_status(case=task)
    state.ml_model_upd_status(case=task)

    print(state.show_calculateble_propreties())
    print()