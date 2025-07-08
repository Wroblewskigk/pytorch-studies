"""
Exercise 3: Search "ways to prevent overfitting in machine learning", write
down 3 of the things you find and a sentence about each. Note: there are
lots of these, so don't worry too much about all of them, just pick 3 and
start with those.
"""

"""
Regularization: Adds a penalty for complexity to the loss function, discouraging 
the model from fitting noise in the training data.

Example: L1 (lasso) and L2 (ridge) regularization are common techniques.

Cross-Validation: Splits the data into multiple train-test sets to ensure the 
model generalizes well to unseen data.

Example: k-fold cross-validation rotates which data is used for validation, helping 
detect overfitting.

Early Stopping: Stops training when performance on a validation set stops improving, 
preventing the model from learning noise in the training data.

Example: Monitor validation loss during training and halt when it starts to increase.
"""