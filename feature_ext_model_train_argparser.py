from feature_extractor import train 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model using Hugging Face Transformers.")

    parser.add_argument("dataframe_path", type=str, help="Local or Remote path to CSV dataset with ['title', 'salary', 'company', 'experience', 'mode', 'skills', 'description'] columns.")
    parser.add_argument("--tokenizer_preset", type=str, default=None, help="Local or Remote path to AutoTokenizer for the model.")
    parser.add_argument("--model_preset", type=str, default=None, help="Local or Remote path to the model.")
    parser.add_argument("--model_revision", type=str, default=None, help="Version of the model to use if remote path is provided.")
    parser.add_argument("--validation_split", type=bool, default=True, help="Whether to use a validation set.")
    parser.add_argument("--metrics", type=bool, default=True, help="Whether to compute metrics during training.")
    parser.add_argument("--training_args", type=str, nargs='*', help="Arguments for training in key=value format.")

    args = parser.parse_args()

    # Convert training_args from list of 'key=value' strings to a dictionary
    training_args = {}
    if args.training_args:
        for arg in args.training_args:
            key, value = arg.split('=')
            try:
                # Attempt to convert the value to a number if possible
                value = eval(value)
            except:
                pass
            training_args[key] = value

    return args.dataframe_path, args.tokenizer_preset, args.model_preset, args.model_revision, args.validation_split, args.metrics, training_args

if __name__ == "__main__":
    dataframe_path, tokenizer_preset, model_preset, model_revision, validation_split, metrics, training_args = parse_args()

    model = train(
        dataframe_path=dataframe_path,
        tokenizer_preset=tokenizer_preset,
        model_preset=model_preset,
        model_revision=model_revision,
        validation_split=validation_split,
        metrics=metrics,
        **training_args
    )

