from neupre.backend.base_backend import BaseBackend


class LstmBackend(BaseBackend):
    def __init__(self, input_dim, hidden_dim_1, output_dim, learning_rate, inp, hid, out, path, mean, std):
        from neupre.misc.builders import build_model_lstm_simple, build_model_recurrent
        super(LstmBackend, self).__init__()

        self.model_multistep = build_model_lstm_simple(input_dim, hidden_dim_1, output_dim)
        self.model_onestep96 = build_model_recurrent(inp, hid, out)
        self.initialize(True, path, mean, std)

    def train(self):
        # log1 = self.model_onestep.fit(self.X_train_onestep, self.y_train_onestep, batch_size=100, nb_epoch=1,
        #                               validation_split=0.1, verbose=1)
        log2 = self.model_multistep.fit(self.X_train_multistep, self.y_train_multistep, batch_size=10, nb_epoch=1,
                                        validation_split=0.1, verbose=1)
        log3 = self.model_onestep96.fit(self.X_train_onestep96, self.y_train_onestep96, batch_size=10, nb_epoch=1,
         validation_split=0.1, verbose=2)
        return [log2, log3]

    def predict(self, X_test_multistep, X_test_onestep96):
        # p1 = self.model_onestep.predict(X_test_onestep)
        p2 = self.model_multistep.predict(X_test_multistep)
        p3 = self.model_onestep96.predict(X_test_onestep96)
        return [p2, p3]
