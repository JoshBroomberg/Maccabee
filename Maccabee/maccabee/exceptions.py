class UnknownEstimandException(Exception):
    def __init__(self):
        super().__init__("Unknown Estimand provided")
