import numpy as np
from cedne.simulator import RateModel
import optuna
from scipy.optimize import minimize as scipy_minimize
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx
import os
import cedne
import getpass
user = getpass.getuser()
optuna.logging.set_verbosity(optuna.logging.WARNING)

CEDNE_ROOT = os.path.dirname(os.path.abspath(cedne.__file__))
PACKAGE_ROOT = CEDNE_ROOT.split('src')[0]

LARGE_LOSS = 1e6

# Postgresql connection
if 'PGUSER' in os.environ:
    PGUSER = os.environ['PGUSER']
else:
    PGUSER = user

if 'PGHOST' in os.environ:
    PGHOST = os.environ['PGHOST']
else:
    PGHOST = 'localhost'

if 'PGPORT' in os.environ:
    PGPORT = os.environ['PGPORT']
else:
    PGPORT = 5432

if 'PGDATABASE' in os.environ:
    PGDATABASE = os.environ['PGDATABASE']
else:
    PGDATABASE = 'cedne_optimization_optuna'
class Optimizer:
    def __init__(self, simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=100, **kwargs):
        """
        Initialize the optimizer.
        
        Args:
            simulation_model (callable): A function that takes parameters and returns simulated data.
            real_data (numpy array): The target time-series data to fit.
            loss_function (callable): A function that calculates the error between simulated and real data.
            parameter_bounds (dict): Bounds for each parameter as {'param1': (min, max), 'param2': (min, max)}.
            vars_to_fit (list): List of variable names to optimize.
            **kwargs: Additional keyword arguments for the optimizer.
        """
        self.simulation_model = simulation_model
        self.loss_function = loss_function
        self.neuron_parameter_bounds = neuron_parameter_bounds
        self.edge_parameter_bounds = edge_parameter_bounds
        self.real_data = np.array([real_data[node] for node in vars_to_fit ])
        self.sim_data = np.zeros(self.real_data.shape)
        self.vars_to_fit = vars_to_fit
        self.num_trials = num_trials

    def optimize(self, initial_guess=None, max_iterations=100):
        """
        Run the optimization process.
        
        Args:
            initial_guess (dict, optional): Initial values for parameters.
            max_iterations (int): Maximum number of optimization iterations.
        
        Returns:
            dict: Optimized parameter values.
        """

        pass

class GradientDescentOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'gradient_descent'

    def optimize(self, max_iterations=100):
        # Implement gradient descent optimization logic
        ## Useful for models with differnetiable losss functions. Perhaps I can use this for the rate model. 
        pass

class GradientFreeOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'gradient_free'

    def optimize(self, max_iterations=100):
        # Implement gradient-free optimization logic
        pass

class ScipyOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'scipy'

    def objective(self, params):
        """
        Objective function for SciPy optimization.
        
        Args:
            params (list): A list of parameter values.
        
        Returns:
            float: Loss value for the given parameter set.
        """
        # Set parameters
        neuron_pars = {}
        edge_pars = {}
        param_index = 0

        if any (np.isnan(params)):
            print("Nan found", params)

        for key, bounds_dict in self.neuron_parameter_bounds.items():
            neuron_pars[key] = {}
            for node in bounds_dict.keys():
                neuron_pars[key][node] = params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            edge_pars[key] = {}
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = params[param_index]
                param_index += 1

        
        self.simulation_model.set_neuron_parameters(neuron_pars)
        self.simulation_model.set_edge_parameters(edge_pars)

        # Run simulation and calculate loss
        simulated_data = self.simulation_model.simulate()

        for j, node in enumerate(self.vars_to_fit):
            self.sim_data[j] = simulated_data[node]
            if any(np.isnan(simulated_data[node])):
                print("Nan found", node.name, simulated_data[node])

        loss = self.loss_function(self.sim_data, self.real_data)
        return loss

    def optimize(self, max_iterations=100):
        """
        Run the optimization process.
        
        Args:
            max_iterations (int): Maximum number of optimization iterations.
        
        Returns:
            dict: Best parameter values.
        """
        # Flatten parameter bounds
        bounds = []
        for bounds_dict in self.neuron_parameter_bounds.values():
            bounds.extend(bounds_dict.values())
        for bounds_dict in self.edge_parameter_bounds.values():
            bounds.extend(bounds_dict.values())

        # Initial guess
        initial_guess = [np.random.uniform(bound[0], bound[1]) for bound in bounds]
        print(len(initial_guess))

        # Run optimization
        result = scipy_minimize(self.objective, initial_guess, bounds=bounds, method='L-BFGS-B', options={'maxiter': self.num_trials, "maxfun": 1000, "gtol": 1e-3, "maxcor": 10})

        print(result)
        # Extract best parameters
        best_params = result.x
        neuron_pars = {key: {} for key in self.neuron_parameter_bounds.keys()}
        edge_pars = {key: {} for key in self.edge_parameter_bounds.keys()}
        param_index = 0

        for key, bounds_dict in self.neuron_parameter_bounds.items():
            for node in bounds_dict.keys():
                neuron_pars[key][node] = best_params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = best_params[param_index]
                param_index += 1

        self.simulation_model.set_neuron_parameters(neuron_pars)
        self.simulation_model.set_edge_parameters(edge_pars)

        return result, self.simulation_model
    
class OptunaOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=100, study_name=None, njobs=None, storage=None, dbtype = 'sqlite', gamma=0.25, **kwargs):
        """
        Initialize the parameter optimizer.
        
        Args:
            simulation_model (SimulationModel): The simulation model to optimize.
            real_data (numpy array): The target time-series data to fit.
            loss_function (callable): A function that calculates the error between simulated and real data.
            parameter_bounds (dict): Bounds for each parameter as {'param_name': (min, max)}.
            vars_to_fit (list): List of variable names to optimize.
            **kwargs: Additional keyword arguments for the optimizer.
        """
        super().__init__(simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials,  **kwargs)
        self.optimization_method = 'optuna'
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=num_trials//3, gamma=gamma)
        self.njobs = njobs
        self.neurons = {}
        self.current_loss = None
        if not study_name:
            self.study_name = 'cedne_optimization_optuna'
        else:
            self.study_name = study_name
        
        if not storage:
            if dbtype == 'sqlite':
                self.storage = f'sqlite:///{PACKAGE_ROOT}/tmp/{self.study_name}/cedne_optimization_optuna.db?timeout=30'
                if not os.path.exists(f'{PACKAGE_ROOT}/tmp/{self.study_name}'):
                    os.makedirs(f'{PACKAGE_ROOT}/tmp/{self.study_name}')
            elif dbtype == 'postgresql':
                storage_link = f"postgresql://{PGUSER}@/{PGDATABASE}?host={PGHOST}&port={PGPORT}"
                # Configure engine options
                engine_kwargs = {
                    "pool_size": self.njobs,       # Match number of parallel jobs
                    "max_overflow": self.njobs,    # Allow extra connections
                    "pool_timeout": 60,        # Wait longer before timeout
                }
                self.storage = optuna.storages.RDBStorage(
                url=storage_link,
                engine_kwargs=engine_kwargs
                )

                #self.storage = f"postgresql://{PGUSER}@{PGHOST}:{PGPORT}/{PGDATABASE}"
        else:
            if dbtype == 'sqlite':
                self.storage = storage
                dbpath = self.storage.split('sqlite:///')[1]
                dbdir = os.path.dirname(dbpath)
                if not os.path.exists(dbdir):
                    os.makedirs(dbdir)
            elif dbtype == 'postgresql':
                self.storage = storage
        print(f"Connecting to Optuna database: {self.storage}")
        try:
            self.study = optuna.create_study(
            study_name= self.study_name,
            storage=self.storage,  # Local file,
            sampler=sampler,
            pruner = optuna.pruners.MedianPruner(),
            direction="minimize",
            load_if_exists=True
            )
            print(f"Created study name: {self.study_name} with storage: {self.storage}.")
        except Exception as e:
            print(f"Optuna failed to connect: {e}")


    def objective(self, trial):
        """
        Objective function for Optuna.
        
        Args:
            trial (optuna.trial.Trial): An Optuna trial object.
        
        Returns:
            float: Loss value for the given parameter set.
        """
        # Suggest parameters within bounds
        # params = {
        #     # key: trial.suggest_uniform(key, *bounds)
        # for key, bounds in self.parameter_bounds.items()
        # }

        # Set parameters
        self.sim_data = np.zeros((len(self.vars_to_fit), len(self.simulation_model.time_points)))

        neuron_pars = {}
        for key, bounds_dict in self.neuron_parameter_bounds.items():
            neuron_pars[key] = {}
            for neuron, bounds in bounds_dict.items():
                neuron_pars[key][neuron] = trial.suggest_float(f'{key}:{str(neuron.name)}', *bounds)
                self.neurons[str(neuron.name)] = neuron

        # print(self.neurons)
        edge_pars = {}
        for key, bounds_dict in self.edge_parameter_bounds.items():
            edge_pars[key] = {}
            for edge, bounds in bounds_dict.items():
                # if key == 'time_constant':
                #     tc = trial.suggest_float(f'{key}:{str(edge[0].name)}:{str(edge[1].name)}', *bounds)
                #     edge_pars[key][edge] = 10**(tc)
                # else:
                edge_pars[key][edge] = trial.suggest_float(f'{key}:{str(edge[0].name)}:{str(edge[1].name)}:{edge[2]}', *bounds)
                self.neurons[str(edge[0].name)] = edge[0]
                self.neurons[str(edge[1].name)] = edge[1]

        self.simulation_model.set_neuron_parameters(neuron_pars)
        self.simulation_model.set_edge_parameters(edge_pars)
        
        # for n, node in self.simulation_model.nodes.items():
        #     print(n, node, node.neuron_parameters)

        # for edge_data in self.simulation_model.edges(data=True):
        #     print(edge_data)        
        

        # Run simulation and calculate loss
        simulated_data = self.simulation_model.simulate()

        if simulated_data is None:
            print(f"Trial {trial.number}: Simulation returned None. Assigning large loss.")
            return LARGE_LOSS
            # trial.report(float("inf"), step=0)
            # raise optuna.TrialPruned()
        
        real_data = np.zeros((len(self.vars_to_fit), len(self.simulation_model.time_points)))
        for j, node in enumerate(self.vars_to_fit):
            self.sim_data[j] = simulated_data[node]
            if np.any(np.isnan(self.sim_data[j])):
                print(f"Trial {trial.number}: NaN found in simulated data for {node.name}. Assigning large loss.")
                return LARGE_LOSS
            # if any(np.isnan(simulated_data[node])):
            #     print("Nan found", node.name, simulated_data[node])
            real_data[j] = self.real_data[j, self.simulation_model.time_points]

        loss = self.loss_function(self.sim_data, real_data)

        trial.report(loss, step=0)
        if np.isnan(loss) or np.isinf(loss):
            print(f"Trial {trial.number}: NaN detected in loss. Assigning large loss.")
            return LARGE_LOSS
            # print("NaN detected in trial. Stopping ")
            # raise optuna.TrialPruned()
        
        self.current_loss = loss
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # print(self.sim_data, self.real_data)
        return loss

    def optimize(self):
        """
        Run the optimization process.
        
        Args:
            n_trials (int): Number of optimization trials.
        
        Returns:
            dict: Best parameter values.
        """
        if self.njobs is None:
            # self.study = optuna.create_study(direction="minimize", storage="sqlite:///optuna.db")
            self.study.optimize(self.objective, n_trials=self.num_trials)
        else:
            self.study = optuna.load_study(study_name=self.study_name, storage=self.storage)
            self.study.optimize(self.objective, n_trials=self.num_trials, n_jobs=self.njobs)
        # Return the best parameter values

        neuron_pars = {key: {} for key in self.neuron_parameter_bounds.keys()}
        edge_pars = {key: {} for key in self.edge_parameter_bounds.keys()}

        if len(self.study.trials) == 0:
            print("No valid trials found. Optimization failed.")
            best_params = None  # Handle empty case
            for trial in self.study.trials:
                print(f"Trial {trial.number} - State: {trial.state}, Value: {trial.value}")
        else:
             best_params = self.study.best_params
        for key, value in best_params.items():
            graph_id_pars = key.split(':')
            if len(graph_id_pars) == 2:
                graph_id_pars[1] = self.neurons[graph_id_pars[1]] # changing for integer nodes
                neuron_pars[graph_id_pars[0]].update({graph_id_pars[1]: value})

            elif len(graph_id_pars) == 4:
                graph_id_pars[1] = self.neurons[graph_id_pars[1]] # changing for integer nodes
                graph_id_pars[2] = self.neurons[graph_id_pars[2]] # changing for integer nodes
                edge_pars[graph_id_pars[0]].update({(graph_id_pars[1], graph_id_pars[2], int(graph_id_pars[3])): value})

        self.simulation_model.set_neuron_parameters(neuron_pars)
        self.simulation_model.set_edge_parameters(edge_pars)
            
        return best_params, self.simulation_model

class BayesianOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, parameter_bounds, vars_to_fit, num_trials=100, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, parameter_bounds, vars_to_fit, num_trials, **kwargs)
        self.optimization_method = 'bayesian'

    def optimize(self, max_iterations=100):
        # Implement Bayesian optimization logic
        pass

class JaxOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, neuron_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'jax'

    @eqx.filter_jit
    def objective(self, params):
        neuron_pars = {}
        edge_pars = {}
        param_index = 0

        for key, bounds_dict in self.neuron_parameter_bounds.items():
            neuron_pars[key] = {}
            for node in bounds_dict.keys():
                neuron_pars[key][node] = params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            edge_pars[key] = {}
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = params[param_index]
                param_index += 1

        self.simulation_model.set_neuron_parameters(neuron_pars)
        self.simulation_model.set_edge_parameters(edge_pars)

        simulated_data = self.simulation_model.simulate()
        loss = self.loss_function(simulated_data, self.real_data)
        return loss

    def optimize(self, max_iterations=100):
        bounds = []
        for bounds_dict in self.neuron_parameter_bounds.values():
            bounds.extend(bounds_dict.values())
        for bounds_dict in self.edge_parameter_bounds.values():
            bounds.extend(bounds_dict.values())

        initial_guess = jnp.array([jnp.mean(bound) for bound in bounds])

        def loss_fn(params):
            return self.objective(params)

        grad_fn = jax.grad(loss_fn)
        params = initial_guess
        for _ in range(max_iterations):
            grads = grad_fn(params)
            params = params - 0.01 * grads

        best_params = params
        neuron_pars = {key: {} for key in self.neuron_parameter_bounds.keys()}
        edge_pars = {key: {} for key in self.edge_parameter_bounds.keys()}
        param_index = 0

        for key, bounds_dict in self.neuron_parameter_bounds.items():
            for node in bounds_dict.keys():
                neuron_pars[key][node] = best_params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = best_params[param_index]
                param_index += 1

        self.simulation_model.set_neuron_parameters(neuron_pars)
        self.simulation_model.set_edge_parameters(edge_pars)

        return best_params, self.simulation_model
    
class BaseVisualizer:
    def __init__(self, optimization_result):
        """
        Initialize the visualizer with the results of the optimization.
        
        Args:
            optimization_result: Results or state object from the optimization library.
        """
        self.optimization_result = optimization_result

    def plot_optimization_history(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def plot_param_importances(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


## Loss functions
def mean_squared_error(simulated, real):
    return ((simulated - real) ** 2).mean()

def correlation_loss(simulated, real):
    return 1 - np.corrcoef(simulated, real)[0, 1]
