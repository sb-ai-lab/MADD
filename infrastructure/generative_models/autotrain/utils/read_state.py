import json 
from pydantic import BaseModel
import os

class BaseState(BaseModel):
    status:str = None
    weights_path:str = None 
    arch_type:str = None
    case_name: str = None
    base_state:dict = {"description" : None,
                       "generative_models": {"data":{"data_path":None,"feature_column":None,"target_column":None,"problem":None},'status':status,'weights_path':weights_path,'arch_type':arch_type,"metric":None},
                     "ml_models":{"data":{"data_path":None,"feature_column":None,"target_column":None,"problem":None},'status':status,'weights_path':weights_path,"metric":None}}
class TrainState:
    """A class representing a state with information about the available trained 
    models for a multi-agent system. The main essence of its operation is the ability to load
      and save a dictionary with a state, using it to analyze the availability of 
      trained (training) generative or predictive models.

    Returns:
        _type_: _description_
    """
    defult_parameters = BaseState()

    def __init__(self,state_path:str=None):
        self.state_path = state_path
        if state_path is not None:
            self.current_state = self.__load_state()
        elif os.path.isfile("automl/state/state.json"):
            self.state_path = "automl/state/state.json"
            self.current_state = self.__load_state()
        else:
            self.state_path = r'automl/new_state.json'
            self.current_state = {}
            self.__save_state()

    def __call__(self, case:str = None, 
                 model:str = None,
                   *args,
                     **kwargs):
        """Return full state if take no arguments.

        Args:
            case (str, optional): Task name. If defined - then return one state of task, named ("case"). Defaults to None.
            model (str, optional): Model state. May be choose from "ml" or "gen" to return only state for generative or ML models.
             Must be specified with argument "case". Defaults to 'ml'.

        Returns:
            dict: State with model information.
        """
        if case is None:
            return self.current_state
        elif self.current_state[case]:
            if model=='ml':
                return self.current_state[case]["ml_models"]
            elif model=='gen':
                return self.current_state[case]["generative_models"]
            else: 
                return self.current_state[case]
        else:
            return None
        
    def add_new_case(self,
                     case_name:str,
                     rewrite=False,
                     description : str = 'Unknown case',
                     **kwargs):
        """Add new case object to state dict with case name and defult parameters. After adding update and save state.

        Args:
            case_name (str): User specified case name.
            rewrite (bool, optional): Indicates whether or not to overwrite an already existing state with this state. Defaults to False.

        Returns:
            None:
        """
        if case_name in self.current_state.keys() and not rewrite:
            print(f'Case already exist! Change name for new case, or use exist case state named {case_name}!')
            print(f"Now using case named '{case_name}'")
            return None
        
        self.current_state[case_name] = self.defult_parameters.base_state
        self.current_state[case_name]['description'] = description
        self.__save_state()

    def gen_model_upd_data(self,
                           case:str,
                           data_path:str=None,
                           feature_column:str = None,
                           target_column:str = None,
                           problem:str = None
                           ):
        """Necessary to update the parameters of a generative model state with the specified name ("case").
          The path to the training data, the type of the problem, and the columns
            with attributes and target parameters in the dataset can be specified.
            All or only any one parameters specified in the function call.

        Args:
            case (str): Case name.
            data_path (str, optional): Path to training data. Defaults to None.
            feature_column (str, optional): Name of training data feature column. Defaults to None.
            target_column (str, optional): Name of training data target column. Defaults to None.
            problem (str, optional): Problem ("regression" or "classification"). Defaults to None.
        """
        if data_path is not None:
            self.current_state[case]["generative_models"]['data']["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["generative_models"]['data']["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["generative_models"]['data']["target_column"] = target_column
        if problem is not None:
            self.current_state[case]["generative_models"]['data']["problem"] = problem
        self.__save_state()

    def ml_model_upd_data(self,
                        case:str,
                        data_path:str=None,
                        feature_column:str = None,
                        target_column:str = None,
                        problem:str = None):
        """Necessary to update the parameters of a ML model state with the specified name ("case").
          The path to the training data, the type of the problem, and the columns
            with attributes and target parameters in the dataset can be specified.
            All or only any one parameters specified in the function call.

        Args:
            case (str): Case name.
            data_path (str, optional): Path to training data. Defaults to None.
            feature_column (str, optional): Name of training data feature column. Defaults to None.
            target_column (str, optional): Name of training data target column. Defaults to None.
            problem (str, optional): Problem ("regression" or "classification"). Defaults to None.
        """
        if data_path is not None:
            self.current_state[case]["ml_models"]['data']["data_path"] = data_path
        if feature_column is not None:
            self.current_state[case]["ml_models"]['data']["feature_column"] = feature_column
        if target_column is not None:
            self.current_state[case]["ml_models"]['data']["target_column"] = target_column
        if problem is not None:
            self.current_state[case]["ml_models"]['data']["problem"] = problem
        self.__save_state()

    def ml_model_upd_status(self,
                            case:str,
                            model_weight_path:str = None,
                            metric = None):
        """Updates the training status of the machine learning model on each call.
          Additionally, you can specify the path to the folder where the model weights are saved after training.
        Also, you can specify the metric value after training.

        Args:
            case (str): Case name.
            model_weight_path (str, optional): Path to trained model weights save folder. Defaults to None.
            metric (_type_, optional): Value of metric after model training. Defaults to None.
        """
        if self.current_state[case]["ml_models"]['status'] is None:
            self.current_state[case]["ml_models"]['status'] = 'Training'
        elif self.current_state[case]["ml_models"]['status'] == 'Training':
            self.current_state[case]["ml_models"]['status'] = 'Trained'
        else:
            print(f'ML model for task "{case}" already trained!')
        if model_weight_path is not None:
            if not os.path.isdir(model_weight_path ):
                    os.mkdir(model_weight_path )
        if self.current_state[case]["ml_models"]['weights_path'] is None and model_weight_path is not None:
            self.current_state[case]["ml_models"]['weights_path'] = model_weight_path 
        if not metric is None:
           self.current_state[case]["ml_models"]['metric'] =  metric
        self.__save_state()

    def gen_model_upd_status(self,
                             case:str,
                             model_weight_path:str = None,
                             metric = None):
        """Updates the training status of the generative model on each call.
          Additionally, you can specify the path to the folder where the model weights are saved after training.
        Also, you can specify the metric value after training.

        Args:
            case (str): Case name.
            model_weight_path (str, optional): Path to trained model weights save folder. Defaults to None.
            metric (_type_, optional): Value of metric after model training. Defaults to None.
        """

        if self.current_state[case]["generative_models"]['status'] is None:
            self.current_state[case]["generative_models"]['status'] = 'Training'
        elif self.current_state[case]["generative_models"]['status'] == 'Training':
            self.current_state[case]["generative_models"]['status'] = 'Trained'
        else:
            print(f'Generative model for task "{case}" already trained!')
        if not os.path.isdir(model_weight_path ):
            os.mkdir(model_weight_path )
        if not self.current_state[case]["generative_models"]['weights_path'] is None and model_weight_path is not None:
            self.current_state[case]["generative_models"]['weights_path'] = model_weight_path
        if not metric is None:
           self.current_state[case]["generative_models"]['metric'] =  metric
        self.__save_state()

    @staticmethod
    def load_state(path:str = r'automl/state.json'):
        return json.load(open(path))
    
    def save(self,path:str = r'automl/state.json'):
        json.dump(self.current_state, open( path, 'w' ) )

    def __save_state(self):
        json.dump(self.current_state, open(self.state_path, 'w' ) )

    def __load_state(self):
        return json.load(open(self.state_path))


if __name__=='__main__':
    task = 'Brain_cancer'
    state = TrainState()
    state.add_new_case(task)
    state.gen_model_upd_data(case=task,data_path='docking_output_files/docking_output_files/data_result.csv')
    state.gen_model_upd_status(case=task)
    state.gen_model_upd_status(case=task)
    state.gen_model_upd_status(case=task)
    state.ml_model_upd_status(case=task)
    state.add_new_case("New task")
    state.save(path= f"automl/state/state.json")
    print()