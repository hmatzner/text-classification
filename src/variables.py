from typing import Dict, Any
import pickle

MAIN_FOLDER = '/Users/hernanmatzner/text_classification/'
VARIABLES_FOLDER = MAIN_FOLDER + 'saved_variables/'


def save_variables(variables: Dict[str, Any]) -> None:
    """
    Saves variables to disk using pickle.

    Parameters:
    - variables: dictionary where the keys are the names to use when saving the variables, and the values are the
    variables to be saved.

    Returns:
    - None
    """

    for variable_name, variable in variables.items():
        with open(f'{VARIABLES_FOLDER}{variable_name}.pickle', 'wb') as f:
            pickle.dump(variable, f)


def read_variable(variable_name: str) -> Any:
    """
    Loads a variable previously saved in disk using pickle.

    Parameters:
    - variable_name: path of the variable saved

    Returns:
    - variable: the loaded variable
    """

    with open(f'{VARIABLES_FOLDER}{variable_name}.pickle', 'rb') as f:
        variable = pickle.load(f)

    return variable


def check_if_exists(variable_name: str):
    """
    Checks if a variable exists in the global scope.

    Parameters:
    - variable_name: name of the variable

    Returns:
    - None
    """

    if variable_name in globals():
        print(f'Variable "{variable_name}" exists.')
    else:
        print(f'Variable "{variable_name}" does not exist.')
