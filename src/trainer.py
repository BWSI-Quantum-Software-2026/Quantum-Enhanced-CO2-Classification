import numpy as np

class QNNTrainer:

    def __init__(
        self,
        model, 
        learning_rate = 0.1, # how large each update to weight is
        batch_size = 10, # number of data samples per batch (days)
        validation_split = 0.1, # what amount of data is used for validation
        epochs = 30 # number of times repeating training, updating epoch
    ):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split

    def train(self, X, Y):
        split_index = int (
            len(X)*(1-self.validation_split)
        )# math to determine which of the dataset is used for training and which is used for validation (first 90% for training and last 10% for validation)

        X_train = X[:split_index]
        Y_train = Y[:split_index]

        X_val = X[split_index:]
        Y_val = Y[split_index:]

        w = np.random.uniform(
            -0.1, 
            0.1, 
            size = len(self.model.weight_params)
        )

        history = {
            'train_loss':[],
            'val_loss':[]
        }

        for epoch in range(self.epochs):
            #get a random, nonrepetitive batch
            index = np.random.choice(
                len(X_train),
                self.batch_size,
                replace = False
            )

            Xb = X_train[index]
            Yb = Y_train[index]

            g = self.model.grad_batch(
                w,
                Xb,
                Yb
            )
            print(np.linalg.norm(g))

            w -= self.learning_rate*g

            train_loss = self.model.batch_loss(
                w,
                X_train,
                Y_train
            )

            val_loss = self.model.batch_loss(
                w,
                X_val,
                Y_val
            )

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            print(
                f"Epoch {epoch}: "
                f"train={train_loss:.6f}, "
                f"val={val_loss:.6f}"
            )

        return w, history
