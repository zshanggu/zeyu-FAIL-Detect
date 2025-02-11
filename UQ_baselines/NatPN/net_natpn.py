from natpn import NaturalPosteriorNetwork
def get_net():
    estimator = NaturalPosteriorNetwork(
        encoder="tabular",
        flow_num_layers=16,
        learning_rate=1e-3,
        learning_rate_decay=True,
        trainer_params=dict(max_epochs=1000),
    )
    return estimator