import json
import os
from . import logger
class Configure(object):
    def __init__(self, config=None, config_json_file=None, _is_top_level=True):
        self.dict = {}
        self._is_top_level_instance = _is_top_level
        loaded_from_file = False
        if config_json_file:
            assert os.path.isfile(config_json_file), f"Error: Configure file not exists! {config_json_file}"
            with open(config_json_file, 'r') as fin:
                loaded_dict = json.load(fin)
            self.dict.update(loaded_dict)
            self.update(loaded_dict)
            loaded_from_file = True
        if config:
            if not loaded_from_file:
                 self.dict.update(config)
            self.update(config)
        if self._is_top_level_instance and self.dict:
            self._log_config()
    def __getitem__(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        elif key in self.dict:
            return self.dict[key]
        else:
            raise KeyError(f"Key '{key}' not found in configuration.")
    def __contains__(self, key):
        return key in self.__dict__
    def add(self, k, v):
        self.__dict__[k] = v
        self.dict[k] = v
    def items(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_') and k != 'dict'}.items()
    def update(self, config):
        assert isinstance(config, dict), "Input config must be a Dictionary!"
        processed_config = {}
        for k, v in config.items():
            if isinstance(v, dict):
                processed_config[k] = Configure(v, _is_top_level=False)
            elif isinstance(v, list):
                processed_config[k] = [Configure(x, _is_top_level=False) if isinstance(x, dict) else x for x in v]
            else:
                processed_config[k] = v
        self.__dict__.update(processed_config)
    def _log_config(self):
        """Logs the configuration structure using a cleaner dict representation."""
        log_dict = self._to_dict()
        try:
            log_str = json.dumps(log_dict, indent=4, default=str)
            logger.info("---- CONFIGURATION START ----")
            for line in log_str.splitlines():
                logger.info(line)
            logger.info("---- CONFIGURATION END ----")
        except TypeError as e:
            logger.error(f"Could not serialize config for logging: {e}")
            logger.info(f"CONFIGURE (raw dict): {log_dict}")
    def _to_dict(self):
        """Recursively convert Configure object back to a dictionary for logging/serialization."""
        d = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_') and k != 'dict':
                if isinstance(v, Configure):
                    d[k] = v._to_dict()
                elif isinstance(v, list):
                    d[k] = [elem._to_dict() if isinstance(elem, Configure) else elem for elem in v]
                else:
                    d[k] = v
        return d
    def get(self, key, default=None):
        return self.__dict__.get(key, default)