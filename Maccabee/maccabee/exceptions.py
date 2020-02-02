class UnknownEstimandException(Exception):
    def __init__(self):
        super().__init__("Unknown Estimand provided")

class UnknownDGPVariableException(Exception):
    def __init__(self):
        super().__init__("Unknown DGP Variable provided")
