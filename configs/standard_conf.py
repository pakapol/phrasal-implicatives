class config:
    vocab_limit = 65536
    vocab_dim = 300
    retrain_embeddings = False
    num_classes = 3
    max_prem_len = 25
    max_hyp_len = 25
    dropout_rate = 0.5
    num_layers = 1
    state_size = 300
    max_grad_norm = 5.
    batch_size = 256
    num_epoch = 2
    data_path = "data/standard"
