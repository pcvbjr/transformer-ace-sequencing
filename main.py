import os

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
#           num_epochs = use early stopping, tracking validation loss per epoch, max 50
#           learning_rate = use addaptive approach, starting with 0.01
#       Sequence Generation Hyperparameters
#           num_decks = 1
#           shuffle_threshold = 0.5
#           sequence_length = 100
#           num_train_sequences = 1e4
#           num_valid_sequences = 1e3


if __name__ == "__main__":

    data_dir = './experiment_data/'

    # Independent Variables
    num_layers = [1, 2, 4]
    
    # Constants
    d_model = 4
    num_heads = 4
    dim_feedforward = 16
    batch_size = 20
    num_epochs = 50
    learning_rate = 0.01
    num_decks = 1
    shuffle_threshold = 0.5
    sequence_length = 100
    num_train_sequences = 10000
    num_valid_sequences = 1000

    constant_args = get_default_args()
    constant_args.update({
        'd_model': d_model,
        'num_heads': num_heads,
        'dim_feedforward': dim_feedforward,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'num_decks': num_decks,
        'shuffle_threshold': shuffle_threshold,
        'sequence_length': sequence_length,
        'num_train_sequences': num_train_sequences,
        'num_valid_sequences': num_valid_sequences,
    })

    # Riffle Only Shuffles
    for num_layers in [1, 2, 4]:
        for num_riffle_passes in [1, 2, 3, 4, 5, 6, 7]:
            print(f'Training RiffleOnlyShuffle with {num_layers} layers and {num_riffle_passes} riffle passes...')
            shuffle_type = 'RiffleOnlyShuffle'
            args = constant_args.copy()
            args.update({
                'shuffle_type': shuffle_type,
                'num_riffle_passes': num_riffle_passes,
                'num_layers': num_layers,
                'output_file': f'{shuffle_type}_{num_riffle_passes}rp_{num_layers}l',
            })
            train(args)