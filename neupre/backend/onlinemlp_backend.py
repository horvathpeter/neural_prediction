from .base_backend import BaseBackend


class Mlp(BaseBackend):
    def __init__(self, input_dim, hidden_dim_1, output_dim, learning_rate):
        from neupre.misc.builders import build_model_mlp
        super(Mlp, self).__init__()
        self.model = build_model_mlp(input_dim, hidden_dim_1, output_dim)
        self.initialize(False)

    def train(self):
        log = self.model.fit(self.X_train, self.y_train, batch_size=1, nb_epoch=20, validation_split=0.1, verbose=2)
        return log

    def predict(self, X_test):
        return self.model.predict(X_test)
