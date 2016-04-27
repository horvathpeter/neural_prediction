from neupre.backend.base_backend import BaseBackend


class Lstm(BaseBackend):
    def __init__(self, input_dim, hidden_dim_1, hidden_dim_2, output_dim, learning_rate):
        from neupre.misc.builders import build_model_lstm
        super(Lstm, self).__init__()
        self.model = build_model_lstm(input_dim, hidden_dim_1, hidden_dim_2, output_dim)
        self.initialize(True)

    def train(self):
        log = self.model.fit(self.X_train, self.y_train, batch_size=100, nb_epoch=10, validation_split=0.1, verbose=2)
        return log

    def predict(self, X_test):
        return self.model.predict(X_test)
