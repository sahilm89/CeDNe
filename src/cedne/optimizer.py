import numpy as np
from cedne.simulator import RateModel
import optuna

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

class GradientFreeOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs):
        super().__init__(simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, **kwargs)
        self.optimization_method = 'gradient_free'

    def optimize(self, max_iterations=100):
        # Implement gradient-free optimization logic
        pass


class OptunaOptimizer(Optimizer):
    def __init__(self, simulation_model, real_data, loss_function, node_parameter_bounds, edge_parameter_bounds, vars_to_fit, num_trials=100, **kwargs):
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
                node_pars[key][node] = trial.suggest_float(f'{key}:{node.label}', *bounds)
        
        edge_pars = {}
        for key, bounds_dict in self.edge_parameter_bounds.items():
            edge_pars[key] = {}
            for edge, bounds in bounds_dict.items():
                edge_pars[key][edge] = trial.suggest_float(f'{key}:{edge[0].label}:{edge[1].label}:{edge[2]}', *bounds)

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
        
        # print(self.sim_data, self.real_data)

        loss = self.loss_function(self.sim_data, self.real_data)
        return loss

    def optimize(self):
        """
        Run the optimization process.
        
        Args:
            n_trials (int): Number of optimization trials.
        
        Returns:
            dict: Best parameter values.
        """
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(self.objective, n_trials=self.num_trials)

        # Return the best parameter values

        node_pars = {key: {} for key in self.node_parameter_bounds.keys()}
        edge_pars = {key: {} for key in self.edge_parameter_bounds.keys()}
        for key, value in self.study.best_params.items():
            graph_id_pars = key.split(':')
            if len(graph_id_pars) == 2:
                node_pars[graph_id_pars[0]].update({self.simulation_model.nodes[int(graph_id_pars[1])]: value})

            elif len(graph_id_pars) == 4:
                edge_pars[graph_id_pars[0]].update({(self.simulation_model.nodes[int(graph_id_pars[1])], self.simulation_model.nodes[int(graph_id_pars[2])], int(graph_id_pars[3])): value})
        
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