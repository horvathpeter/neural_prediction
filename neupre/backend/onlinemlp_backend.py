from .base_backend import BaseBackend


class MlpBackend(BaseBackend):
    def __init__(self, inpmulti, hidmulti, outmulti, learning_rate, inp96, hid96, out96, path, buffsize, mean, std, statspath):
        from neupre.misc.builders import build_model_mlp
        super(MlpBackend, self).__init__(int(buffsize))

        self.model_multistep = build_model_mlp(inpmulti, hidmulti, outmulti)
        self.model_onestep96 = build_model_mlp(inp96, hid96, out96)
        self.initialize(False, path, mean, std, statspath)

    def train(self):
        log2 = self.model_multistep.fit(self.X_train_multistep, self.y_train_multistep, batch_size=10, nb_epoch=2,
                                        validation_split=0.1, verbose=1)
        log3 = self.model_onestep96.fit(self.X_train_onestep96, self.y_train_onestep96, batch_size=10, nb_epoch=2,
                                        validation_split=0.1, verbose=1)
        return [log2, log3]

    def predict(self, X_test_multistep, X_test_onestep96):
        p2 = self.model_multistep.predict(X_test_multistep)
        p3 = self.model_onestep96.predict(X_test_onestep96)
        return [p2, p3]
