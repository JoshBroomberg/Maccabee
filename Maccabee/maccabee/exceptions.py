class UnknownEstimandException(Exception):
    def __init__(self):
        super().__init__("Unknown Estimand provided")

class UnknownDGPVariableException(Exception):
    def __init__(self):
        super().__init__("Unknown DGP Variable provided")

class DGPVariableMissingException(Exception):
    def __init__(self, msg):
        super().__init__(msg)
