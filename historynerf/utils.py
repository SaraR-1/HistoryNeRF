from typing import List, Tuple, Type, Any
from hydra.core.config_store import ConfigStore

def register_configs(config_list: List[Tuple[str, str, Type[Any]]]) -> None:
    """
    Register configurations in the Hydra Config Store.
    
    Parameters:
        config_list: List of tuples containing (group_name, config_name, config_node).
                     group_name: Name of the configuration group
                     config_name: Specific name of the configuration
                     config_node: Class type of the configuration node
    """
    cs = ConfigStore.instance()
    for group_name, config_name, config_node in config_list:
        if group_name == 'base':
            cs.store(name=config_name, node=config_node)
        else:
            cs.store(group=f"{group_name}", name=config_name, node=config_node)
