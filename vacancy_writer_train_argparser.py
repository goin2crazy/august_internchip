import argparse
from texts_writer import train

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")

    # Define arguments
    parser.add_argument('--dataframe_path', type=str, required=True, help="Local or Remote path to CSV dataset.")
    parser.add_argument('--tokenizer_preset', type=str, required=True, help="Local or Remote path to AutoTokenizer.")
    parser.add_argument('--model_preset', type=str, required=True, help="Local or Remote path to model.")
    parser.add_argument('--model_revision', type=str, default=None, help="Version of the model (if using remote path).")
    parser.add_argument('--validation_split', action='store_true', help="Use a validation set for model validation.")
    
    # Additional arguments to be passed as training_args
    parser.add_argument('--training_args', nargs='*', help="Additional training arguments in key=value format.")

    # Parse arguments
    args = parser.parse_args()

    # Convert training_args to dictionary
    if args.training_args:
        training_args = dict(arg.split('=') for arg in args.training_args)
    else:
        training_args = {}

    # Call the train function with parsed arguments
    train(
        dataframe_path=args.dataframe_path,
        tokenizer_preset=args.tokenizer_preset,
        model_preset=args.model_preset,
        model_revision=args.model_revision,
        validation_split=args.validation_split,
        **training_args
    )

if __name__ == "__main__":
    parse_args()
