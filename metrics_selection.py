class Metrics:
    def __init__(self):
        self.available_metrics = [
            "confusionMatrix",
            "distance",
            "HCE",
            "avgPrecision",
            "MCC",
            "ROCAUC",
            "clDice",
            "graphDistances",
            "morph"
        ]
        self.states = {name: True for name in self.available_metrics}

    def enable_all(self):
        for name in self.states:
            self.states[name] = True

    def disable_all(self):
        for name in self.states:
            self.states[name] = False

    def enable(self, names):
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name in self.states:
                self.states[name] = True
            else:
                print(f"Warning: Metric '{name}' not found in configuration.")

    def disable(self, names):
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name in self.states:
                self.states[name] = False

    def is_active(self, name):
        return self.states.get(name, False)

    def __repr__(self):
        status = [f"{k}: {'ON' if v else 'OFF'}" for k, v in self.states.items()]
        return f"<Metrics Config> {status}"