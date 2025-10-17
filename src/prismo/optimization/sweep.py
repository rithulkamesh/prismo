"""
Parameter sweep framework for batch simulations.

This module provides tools for running parameter sweeps, exploring design
spaces, and aggregating results from multiple simulations.
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from tqdm import tqdm


@dataclass
class SweepParameter:
    """
    Definition of a sweep parameter.

    Attributes
    ----------
    name : str
        Parameter name.
    values : list or array
        Values to sweep over.
    unit : str, optional
        Physical unit for documentation.
    """

    name: str
    values: list
    unit: str = ""

    def __len__(self) -> int:
        return len(self.values)


class ParameterSweep:
    """
    Parameter sweep executor for batch simulations.

    Runs multiple simulations with varying parameters and aggregates results.
    Supports parallel execution and result caching.

    Parameters
    ----------
    parameters : List[SweepParameter]
        Parameters to sweep over.
    simulation_func : Callable
        Function that runs simulation: func(params_dict) -> results_dict
    output_dir : Path, optional
        Directory for saving results.
    parallel : bool, optional
        Whether to run simulations in parallel, default=False.
    num_workers : int, optional
        Number of parallel workers. Default is number of CPU cores.
    """

    def __init__(
        self,
        parameters: List[SweepParameter],
        simulation_func: Callable[[Dict[str, Any]], Dict[str, Any]],
        output_dir: Optional[Path] = None,
        parallel: bool = False,
        num_workers: Optional[int] = None,
    ):
        self.parameters = parameters
        self.simulation_func = simulation_func
        self.output_dir = Path(output_dir) if output_dir else Path("./sweep_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.parallel = parallel
        self.num_workers = num_workers

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self.parameter_combinations: List[Dict[str, Any]] = []

        # Generate parameter combinations
        self._generate_combinations()

    def _generate_combinations(self) -> None:
        """Generate all parameter combinations for the sweep."""
        if len(self.parameters) == 0:
            return

        # Create meshgrid of all parameter values
        param_grids = np.meshgrid(*[p.values for p in self.parameters], indexing="ij")

        # Flatten and create dictionaries
        n_combinations = param_grids[0].size

        for i in range(n_combinations):
            combo = {}
            for param_idx, param in enumerate(self.parameters):
                value = param_grids[param_idx].flat[i]
                combo[param.name] = value

            self.parameter_combinations.append(combo)

    def run(self, show_progress: bool = True) -> List[Dict[str, Any]]:
        """
        Execute parameter sweep.

        Parameters
        ----------
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        List[dict]
            Results for each parameter combination.
        """
        n_total = len(self.parameter_combinations)

        if self.parallel:
            # Parallel execution
            results = self._run_parallel(show_progress)
        else:
            # Sequential execution
            results = self._run_sequential(show_progress)

        self.results = results
        return results

    def _run_sequential(self, show_progress: bool) -> List[Dict[str, Any]]:
        """Run simulations sequentially."""
        results = []

        iterator = self.parameter_combinations
        if show_progress:
            iterator = tqdm(iterator, desc="Parameter Sweep")

        for params in iterator:
            result = self._run_single_simulation(params)
            results.append(result)

        return results

    def _run_parallel(self, show_progress: bool) -> List[Dict[str, Any]]:
        """Run simulations in parallel."""
        with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [
                executor.submit(self._run_single_simulation, params)
                for params in self.parameter_combinations
            ]

            results = []
            iterator = futures
            if show_progress:
                from concurrent.futures import as_completed

                iterator = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Parameter Sweep (Parallel)",
                )

            for future in iterator:
                result = future.result()
                results.append(result)

        return results

    def _run_single_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single simulation with given parameters.

        Parameters
        ----------
        params : dict
            Parameter dictionary.

        Returns
        -------
        dict
            Simulation results.
        """
        try:
            # Run simulation
            result = self.simulation_func(params)

            # Add parameters to result
            result["parameters"] = params
            result["status"] = "success"

            return result

        except Exception as e:
            # Handle errors gracefully
            return {"parameters": params, "status": "error", "error": str(e)}

    def save_results(self, filename: str = "sweep_results.json") -> Path:
        """
        Save sweep results to JSON file.

        Parameters
        ----------
        filename : str
            Output filename.

        Returns
        -------
        Path
            Path to saved file.
        """
        output_path = self.output_dir / filename

        # Convert numpy arrays to lists for JSON serialization
        results_serializable = []
        for result in self.results:
            result_copy = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    result_copy[key] = value.tolist()
                else:
                    result_copy[key] = value
            results_serializable.append(result_copy)

        with open(output_path, "w") as f:
            json.dump(results_serializable, f, indent=2)

        return output_path

    def get_result_array(self, result_key: str) -> np.ndarray:
        """
        Extract a specific result as an array.

        Parameters
        ----------
        result_key : str
            Key in result dictionary.

        Returns
        -------
        ndarray
            Results reshaped according to parameter dimensions.
        """
        # Extract values
        values = [r.get(result_key, np.nan) for r in self.results]

        # Reshape according to parameter dimensions
        shape = tuple(len(p.values) for p in self.parameters)

        return np.array(values).reshape(shape)

    def find_optimal(
        self, metric: str, maximize: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Find parameter combination with optimal metric.

        Parameters
        ----------
        metric : str
            Metric key to optimize.
        maximize : bool
            Whether to maximize (True) or minimize (False).

        Returns
        -------
        Tuple[dict, dict]
            (optimal_parameters, optimal_results)
        """
        metric_values = [r.get(metric, np.nan) for r in self.results]

        if maximize:
            best_idx = np.nanargmax(metric_values)
        else:
            best_idx = np.nanargmin(metric_values)

        optimal_result = self.results[best_idx]
        optimal_params = optimal_result["parameters"]

        return optimal_params, optimal_result

    def plot_sweep_1d(
        self, x_param: str, y_metrics: List[str], save_path: Optional[Path] = None
    ) -> None:
        """
        Plot 1D parameter sweep results.

        Parameters
        ----------
        x_param : str
            Parameter name for x-axis.
        y_metrics : List[str]
            Metric names for y-axis.
        save_path : Path, optional
            Path to save figure.
        """
        import matplotlib.pyplot as plt

        # Find parameter index
        param_idx = next(i for i, p in enumerate(self.parameters) if p.name == x_param)
        x_values = self.parameters[param_idx].values

        fig, ax = plt.subplots(figsize=(10, 6))

        for metric in y_metrics:
            y_values = [r.get(metric, np.nan) for r in self.results]
            ax.plot(x_values, y_values, marker="o", label=metric)

        ax.set_xlabel(f"{x_param} [{self.parameters[param_idx].unit}]")
        ax.set_ylabel("Metric Value")
        ax.legend()
        ax.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def plot_sweep_2d(
        self, x_param: str, y_param: str, metric: str, save_path: Optional[Path] = None
    ) -> None:
        """
        Plot 2D parameter sweep results as heatmap.

        Parameters
        ----------
        x_param, y_param : str
            Parameter names for axes.
        metric : str
            Metric to visualize.
        save_path : Path, optional
            Path to save figure.
        """
        import matplotlib.pyplot as plt

        # Get parameter indices
        x_idx = next(i for i, p in enumerate(self.parameters) if p.name == x_param)
        y_idx = next(i for i, p in enumerate(self.parameters) if p.name == y_param)

        x_values = self.parameters[x_idx].values
        y_values = self.parameters[y_idx].values

        # Extract metric as 2D array
        metric_array = self.get_result_array(metric)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        im = ax.imshow(
            metric_array.T,
            aspect="auto",
            origin="lower",
            extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
            cmap="viridis",
        )

        ax.set_xlabel(f"{x_param} [{self.parameters[x_idx].unit}]")
        ax.set_ylabel(f"{y_param} [{self.parameters[y_idx].unit}]")
        ax.set_title(f"{metric}")

        plt.colorbar(im, ax=ax)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def __repr__(self) -> str:
        """String representation."""
        n_combinations = len(self.parameter_combinations)
        param_names = [p.name for p in self.parameters]
        return (
            f"ParameterSweep(parameters={param_names}, "
            f"combinations={n_combinations}, "
            f"completed={len(self.results)})"
        )
