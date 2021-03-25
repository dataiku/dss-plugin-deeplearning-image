class DkuApplication(object):
    def __init__(self, name, label, source, model_func, preprocessing, weights, input_shape=None):
        self.name = name
        self.label = label
        self.source = source
        self.model_func = model_func
        self.preprocessing = preprocessing
        self.input_shape = input_shape
        self.weights = weights
        self.model = None

    def is_keras_application(self):
        return self.source == "keras"

    def get_weights_url(self, trained_on):
        assert trained_on in self.weights, "You provided a wrong field 'trained_on'. Avilable are {}.".format(
            str(self.weights.keys())
        )
        return self.weights.get(trained_on)

    def jsonify(self):
        return self.name.value
