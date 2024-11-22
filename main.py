from models.train import train, get_default_args

# Experimental design:
#   Independent Variables (take each combination):
#       num_layers: number of TransformerEncoderLayers in transformer model
#           - values: 1, 2, 4
#       shuffle_type: type of shuffle to use
#           - test 1 (scientific) values: RiffleOnlyShuffle()
#               - num_riffle_passes = [1, 2, 3, 4, 5, 6, 7]
#           - test 2 (application) values: HomeShuffle(), CasinoHandShuffle(), AutomaticMachineShuffle()
#   Dependent Variable:
#       validation loss (negative log-likelihood loss)
#   Constants:
#       Model Architecture Hyperparameters:
#           d_model = 4
#           num_heads = 4
#           dim_feedforward = 16
#       Training Hyperparameters
#           batch_size = *** max out out GPU
#           num_epochs = *** need to explore - use early stopping, tracking validation loss per epoch
#           learning_rate = *** need to explore - use addaptive approach
#       Sequence Generation Hyperparameters
#           num_decks = 1
#           shuffle_threshold = 0.5
#           sequence_length = 256
#           num_train_sequences = 1e4
#           num_valid_sequences = 1e3


if __name__ == "__main__":
    