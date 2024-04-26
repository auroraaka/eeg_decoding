import yaml
import copy

class Config:
    def __init__(self, filepath):
        self.filepath = filepath
        self.__dict__['data'] = self.load_config(filepath)
    
    def load_config(self, filepath):
        with open(filepath, 'r') as file:
            data = yaml.safe_load(file)
        return self.parse_config(data)
    
    def parse_config(self, data):
        if isinstance(data, dict):
            return ConfigDict(**data)
        return data

    def __getattr__(self, name):
        return getattr(self.data, name)
    
    def __setattr__(self, name, value):
        if name == "filepath" or name == "data":
            object.__setattr__(self, name, value)
        else:
            setattr(self.data, name, value)

    def deepcopy(self):
        return Config(self.filepath)
    
    def save(self):
        with open(self.filepath, 'w') as file:
            yaml.safe_dump(self.data.to_dict(), file)

class ConfigDict:
    def __init__(self, **entries):
        for key, value in entries.items():
            if isinstance(value, dict):
                self.__dict__[key] = ConfigDict(**value)
            else:
                self.__dict__[key] = value
    
    def __getattr__(self, name):
        if name not in self.__dict__:
            raise AttributeError(f"Attribute '{name}' not found in configuration.")
        return self.__dict__[name]
    
    def __setattr__(self, name, value):
        super().__setattr__(name, value)

    def deepcopy(self):
        entries_copy = copy.deepcopy({k: v for k, v in self.__dict__.items()})
        return ConfigDict(**entries_copy)
    
    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, ConfigDict) else v for k, v in self.__dict__.items()}
    
def update_config(auto_config, custom_config_data):
    for key, value in custom_config_data.items():
        if isinstance(value, ConfigDict):
            if not hasattr(auto_config, key):
                setattr(auto_config, key, type(auto_config)())
            update_config(getattr(auto_config, key), value.__dict__)
        else:
            setattr(auto_config, key, value)