"""
Exercise 2: Use Python's argparse module to be able to send the train.py custom hyperparameter
values for training procedures.

- Add an argument for using a different:
        Training/testing directory
        Learning rate
        Batch size
        Number of epochs to train for
        Number of hidden units in the TinyVGG model
- Keep the default values for each of the above arguments as what they already are (as in
notebook 05).
- For example, you should be able to run something similar to the following line to train a
TinyVGG model with a learning rate of 0.003 and a batch size of 64 for 20 epochs:
python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20.
- Note: Since train.py leverages the other scripts we created in section 05, such as,
model_builder.py, utils.py and engine.py, you'll have to make sure they're available to use too.
You can find these in the going_modular folder on the course GitHub:
https://github.com/mrdbourke/pytorch-deep-learning/
"""

import argparse

parser = argparse.ArgumentParser(description="Train a TinyVGG image classifier with custom "
                                             "hyperparameters.")

parser.add_argument("--train_dir",
                    type=str,
                    default="data/pizza_steak_sushi/train",
                    help="Path to training data directory")

parser.add_argument("--test_dir",
                    type=str,
                    default="data/pizza_steak_sushi/test",
                    help="Path to testing data directory")

parser.add_argument("--learning_rate",
                    type=float,
                    default=0.001,
                    help="Learning rate for optimizer")

parser.add_argument("--batch_size",
                    type=int,
                    default=32,
                    help="Batch size for DataLoader")

parser.add_argument("--num_epochs",
                    type=int,
                    default=5,
                    help="Number of training epochs")

parser.add_argument("--hidden_units",
                    type=int,
                    default=10,
                    help="Number of hidden units in TinyVGG")

args = parser.parse_args()

# Example usage in your script:
# train_dir = args.train_dir
# test_dir = args.test_dir
# learning_rate = args.learning_rate
# batch_size = args.batch_size
# num_epochs = args.num_epochs
# hidden_units = args.hidden_units

# Example usage in CLI
# python Z02.py --learning_rate 0.003 --batch_size 64 --num_epochs 20
# python train.py --train_dir "data/my_custom_train"
#                 --test_dir "data/my_custom_test"
#                 --hidden_units 32

# If you omit any argument, the script will use the default value.
