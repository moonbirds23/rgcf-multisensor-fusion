"""Small compatibility registry used by older scripts."""

_SCENARIOS = {}
_MODELS = {}
_FAULTS = {}


def register_scenario(name, obj): _SCENARIOS[name] = obj
def register_model(name, obj): _MODELS[name] = obj
def register_fault(name, obj): _FAULTS[name] = obj
def get_registered_scenario_names(): return sorted(_SCENARIOS)
def get_registered_model_names(): return sorted(_MODELS)
def get_registered_fault_names(): return sorted(_FAULTS)
def get_scenario_registry(): return dict(_SCENARIOS)
def get_model_registry(): return dict(_MODELS)
def get_fault_registry(): return dict(_FAULTS)
def get_scenario(name): return _SCENARIOS[name]
def get_model(name): return _MODELS[name]
def get_fault(name): return _FAULTS[name]
