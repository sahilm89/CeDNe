import numpy as np
from cedne.simulator import RateModel
import optuna
from scipy.optimize import minimize as scipy_minimize
import jax
import jax.numpy as jnp
import diffrax as dfx
import equinox as eqx

optuna.logging.set_verbosity(optuna.logging.WARNING)

class Optimizer:
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=100, **kwargs):
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
        self.node_parameter_bounds = node_parameter_bounds
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
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'gradient_descent'

    def optimize(self, max_iterations=100):
        # Implement gradient descent optimization logic
        ## Useful for models with differnetiable losss functions. Perhaps I can use this for the rate model. 
        pass

class GradientFreeOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'gradient_free'

    def optimize(self, max_iterations=100):
        # Implement gradient-free optimization logic
        pass

class ScipyOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
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
        node_pars = {}
        edge_pars = {}
        param_index = 0

        if any (np.isnan(params)):
            print("Nan found", params)

        for key, bounds_dict in self.node_parameter_bounds.items():
            node_pars[key] = {}
            for node in bounds_dict.keys():
                node_pars[key][node] = params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            edge_pars[key] = {}
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = params[param_index]
                param_index += 1

        
        self.simulation_model.set_node_parameters(node_pars)
        self.simulation_model.set_edge_parameters(edge_pars)

        # Run simulation and calculate loss
        simulated_data = self.simulation_model.simulate()

        for j, node in enumerate(self.vars_to_fit):
            self.sim_data[j] = simulated_data[node]
            if any(np.isnan(simulated_data[node])):
                print("Nan found", node.label, simulated_data[node])

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
        for bounds_dict in self.node_parameter_bounds.values():
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
        node_pars = {key: {} for key in self.node_parameter_bounds.keys()}
        edge_pars = {key: {} for key in self.edge_parameter_bounds.keys()}
        param_index = 0

        for key, bounds_dict in self.node_parameter_bounds.items():
            for node in bounds_dict.keys():
                node_pars[key][node] = best_params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = best_params[param_index]
                param_index += 1

        self.simulation_model.set_node_parameters(node_pars)
        self.simulation_model.set_edge_parameters(edge_pars)

        return result, self.simulation_model
    
class OptunaOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=100, num_workers=None, **kwargs):
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
        super().__init__(simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials, **kwargs)
        self.optimization_method = 'optuna'
        self.study = None
        self.node_dict = {}
        self.num_workers = num_workers

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
        
        node_pars = {}
        for key, bounds_dict in self.node_parameter_bounds.items():
            node_pars[key] = {}
            for node, bounds in bounds_dict.items():
                node_pars[key][node] = trial.suggest_float(f'{key}:{str(node.label)}', *bounds)
                self.node_dict[str(node.label)] = node

        # print(self.node_dict)
        edge_pars = {}
        for key, bounds_dict in self.edge_parameter_bounds.items():
            edge_pars[key] = {}
            for edge, bounds in bounds_dict.items():
                # if key == 'time_constant':
                #     tc = trial.suggest_float(f'{key}:{str(edge[0].label)}:{str(edge[1].label)}', *bounds)
                #     edge_pars[key][edge] = 10**(tc)
                # else:
                edge_pars[key][edge] = trial.suggest_float(f'{key}:{str(edge[0].label)}:{str(edge[1].label)}:{edge[2]}', *bounds)
                self.node_dict[str(edge[0].label)] = edge[0]
                self.node_dict[str(edge[1].label)] = edge[1]

        self.simulation_model.set_node_parameters(node_pars)
        self.simulation_model.set_edge_parameters(edge_pars)
        
        # for n, node in self.simulation_model.nodes.items():
        #     print(n, node, node.node_parameters)

        # for edge_data in self.simulation_model.edges(data=True):
        #     print(edge_data)        
        

        # Run simulation and calculate loss
        simulated_data = self.simulation_model.simulate()

        for j, node in enumerate(self.vars_to_fit):
            self.sim_data[j] = simulated_data[node]
            if any(np.isnan(simulated_data[node])):
                print("Nan found", node.label, simulated_data[node])
        
        # print(self.sim_data, self.real_data)

        loss = self.loss_function(self.sim_data, self.real_data)
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
        self.study = optuna.create_study(study_name="optuna", direction="minimize", storage="sqlite:///optuna.db", sampler=optuna.samplers.TPESampler(self.num_workers))
        self.study.optimize(self.objective, n_trials=self.num_trials)
        if self.num_workers is not None:
            self.study = self.study.to_parallel(self.num_trials, self.num_workers)
        # Return the best parameter values

        node_pars = {key: {} for key in self.node_parameter_bounds.keys()}
        edge_pars = {key: {} for key in self.edge_parameter_bounds.keys()}
        for key, value in self.study.best_params.items():
            graph_id_pars = key.split(':')
            if len(graph_id_pars) == 2:
                graph_id_pars[1] = self.node_dict[graph_id_pars[1]] # changing for integer nodes
                node_pars[graph_id_pars[0]].update({graph_id_pars[1]: value})

            elif len(graph_id_pars) == 4:
                graph_id_pars[1] = self.node_dict[graph_id_pars[1]] # changing for integer nodes
                graph_id_pars[2] = self.node_dict[graph_id_pars[2]] # changing for integer nodes
                edge_pars[graph_id_pars[0]].update({(graph_id_pars[1], graph_id_pars[2], int(graph_id_pars[3])): value})
        
        self.simulation_model.set_node_parameters(node_pars)
        self.simulation_model.set_edge_parameters(edge_pars)
            
        return self.study.best_params, self.simulation_model

class BayesianOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, parameter_bounds, vars_to_fit, num_trials=100, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, parameter_bounds, vars_to_fit, num_trials, **kwargs)
        self.optimization_method = 'bayesian'

    def optimize(self, max_iterations=100):
        # Implement Bayesian optimization logic
        pass

class JaxOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'jax'

    @eqx.filter_jit
    def objective(self, params):
        node_pars = {}
        edge_pars = {}
        param_index = 0

        for key, bounds_dict in self.node_parameter_bounds.items():
            node_pars[key] = {}
            for node in bounds_dict.keys():
                node_pars[key][node] = params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            edge_pars[key] = {}
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = params[param_index]
                param_index += 1

        self.simulation_model.set_node_parameters(node_pars)
        self.simulation_model.set_edge_parameters(edge_pars)

        simulated_data = self.simulation_model.simulate()
        loss = self.loss_function(simulated_data, self.real_data)
        return loss

    def optimize(self, max_iterations=100):
        bounds = []
        for bounds_dict in self.node_parameter_bounds.values():
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
        node_pars = {key: {} for key in self.node_parameter_bounds.keys()}
        edge_pars = {key: {} for key in self.edge_parameter_bounds.keys()}
        param_index = 0

        for key, bounds_dict in self.node_parameter_bounds.items():
            for node in bounds_dict.keys():
                node_pars[key][node] = best_params[param_index]
                param_index += 1

        for key, bounds_dict in self.edge_parameter_bounds.items():
            for edge in bounds_dict.keys():
                edge_pars[key][edge] = best_params[param_index]
                param_index += 1

        self.simulation_model.set_node_parameters(node_pars)
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