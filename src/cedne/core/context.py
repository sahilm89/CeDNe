'''
This file keeps track of the various contexts that can be used in
experiments. Each context is a dictionary of parameters that can be
used to define the experimental conditions. The contexts are
defined as classes, and each class has a name and a description.
The classes are:
- ContextBase: The base class for all contexts.
- StimulusContext: A context that defines the stimulus parameters.
- EnvironmentalContext: A context that defines the environmental parameters.
- InternalStateContext: A context that defines the internal state parameters.
- ExperimentalContext: A context that defines the experimental parameters.
- Context: A context that combines all the above contexts.
'''

__author__ = "Sahil Moza"
__date__ = "2025-04-09"
__license__ = "MIT"


class ContextBase:
    ''' Base class for all contexts. '''
    def __init__(self, name, description=None):
        '''
        Initializes the context with the given name and description.
        '''
        self.name = name
        self.description = description or ""
        self.network = None
        self.notes = {}

class StimulusContext(ContextBase):
    ''' Context for defining stimulus parameters. '''
    def __init__(self, name, stimulus_params, **kwargs):
        super().__init__(name, **kwargs)
        self.stimuli = stimulus_params  # Dict

class EnvironmentalContext(ContextBase):
    ''' Context for defining environmental parameters. '''
    def __init__(self, name, environment_params, **kwargs):
        super().__init__(name, **kwargs)
        self.environment = environment_params  # Dict

class InternalStateContext(ContextBase):
    ''' Context for defining internal state parameters. '''
    def __init__(self, name, internal_state_params, **kwargs):
        super().__init__(name, **kwargs)
        self.internal_state = internal_state_params

class ExperimentalContext(ContextBase):
    def __init__(self, name, experimental_params, **kwargs):
        '''
        Initializes the experimental context with the given name and
        experimental parameters.
        '''
        super().__init__(name, **kwargs)
        self.experimental_conditions = experimental_params

class Context(ContextBase):
    ''' Context that combines all the above contexts. '''
    def __init__(
        self,
        name,
        stimulus=None,
        environment=None,
        internal=None,
        experimental=None,
        description=None
    ):
        super().__init__(name, description=description)
        self.stimulus = stimulus  # StimulusContext
        self.environment = environment  # EnvironmentalContext
        self.internal = internal  # InternalStateContext
        self.experimental = experimental  # ExperimentalContext

        self.network = None

    def get_signature_dict(self):
        ''' Returns a dictionary of all the parameters in the context. '''
        sig = {}
        for context in [self.stimulus, self.environment, self.internal, self.experimental]:
            if context:
                sig.update({
                    f"{context.__class__.__name__}.{k}": v
                    for k, v in vars(context).items()
                    if not k.startswith("_") and k not in {"name", "description"}
                })
        return sig

    def get_signature_string(self):
        ''' Returns a string representation of the context signature. '''
        sig = self.get_signature_dict()
        return "_".join(f"{k}:{v}" for k, v in sorted(sig.items()))