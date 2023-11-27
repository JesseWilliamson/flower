import argparse
import flwr as fl
import utils
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression
from typing import Dict


def fit_round(server_round: int) -> Dict:
    """Send round number to client."""
    return {"server_round": server_round}


def get_evaluate_fn(model: LogisticRegression):
    """Return an evaluation function for server-side evaluation."""

    # Load test data here to avoid the overhead of doing it in `evaluate` itself
    _, (X_test, y_test) = utils.load_mnist()

    # The `evaluate` function will be called after every round
    def evaluate(server_round, parameters: fl.common.NDArrays, config):
        # Update model with the latest parameters
        utils.set_model_params(model, parameters)
        loss = log_loss(y_test, model.predict_proba(X_test))
        accuracy = model.score(X_test, y_test)
        return loss, {"accuracy": accuracy}

    return evaluate


# Start Flower server for five rounds of federated learning
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--num_rounds", "-c", type=int, default=5, help="Number of rounds of training"
    )
    args = parser.parse_args()

    model = LogisticRegression()
    utils.set_initial_params(model)
    strategy = fl.server.strategy.FedAvg(
        min_available_clients=2,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_round,
    )
    fl_results = fl.server.start_server(
        server_address="0.0.0.0:8080",
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
    )

    # Print without formatting to make data easier to copy into a spreadsheet
    print("Loss, distributed")
    for entry in fl_results.losses_distributed:
        print(entry[1])
    print("Loss, centralized")
    for entry in fl_results.losses_centralized:
        print(entry[1])
    print("Accuracy, centralized")
    for entry in fl_results.metrics_centralized["accuracy"]:
        print(entry[1])

    # Definition of History class from src/py/flwr/server/history.py:
    # self.losses_distributed: List[Tuple[int, float]] = []
    # self.losses_centralized: List[Tuple[int, float]] = []
    # self.metrics_distributed_fit: Dict[str, List[Tuple[int, Scalar]]] = {}
    # self.metrics_distributed: Dict[str, List[Tuple[int, Scalar]]] = {}
    # self.metrics_centralized: Dict[str, List[Tuple[int, Scalar]]] = {}
