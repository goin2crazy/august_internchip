import argparse
from texts_writer import train
from web_collecting.parse_data import run

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

    # Define arguments
    parser.add_argument('--tokenizer_preset', type=str, required=True, help="Local or Remote path to AutoTokenizer.")
    parser.add_argument('--model_preset', type=str, required=True, help="Local or Remote path to model.")
    parser.add_argument('--model_revision', type=str, default=None, help="Version of the model (if using remote path).")

    # Additional arguments to be passed as training_args
    parser.add_argument('--training_args', nargs='*', help="Additional training arguments in key=value format.")

        
    parser.add_argument(
        '--collecting-save-folder',
        type=str,
        default='',
        help='Folder where the collected data will be saved.'
    )

    parser.add_argument(
        '--collecting-collector-verbose',
        type=int,
        default=0,
        help='Verbosity level for the scraper (0 for silent, 1 for verbose).'
    )

    parser.add_argument(
        '--collecting-hub-path',
        type=str,
        default='doublecringe123/parsed-hh-last-tree-days-collection',
        help='Path to the Hugging Face hub where the dataset will be pushed.'
    )


    # Parse arguments
    args = parser.parse_args()

    data_path =  run(
        save_folder=args.collecting_save_folder,
        collector_verbose=args.collecting_collector_verbose,
        hub_path=args.collecting_hub_path
    )

    # Convert training_args to dictionary
    if args.training_args:
        training_args = dict(arg.split('=') for arg in args.training_args)
    else:
        training_args = {}

    training_args['num_train_epochs'] = 2 
                # optimizer 
    training_args['weight_decay'] = 1e-4, 
    training_args['learning_rate'] = 6e-4, 

    # Call the train function with parsed arguments
    train(
        dataframe_path=data_path,
        tokenizer_preset=args.tokenizer_preset,
        model_preset=args.model_preset,
        model_revision=args.model_revision,
        validation_split=False,
        **training_args
    )

if __name__ == "__main__":
    parse_args()
