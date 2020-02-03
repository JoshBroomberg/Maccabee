class UnknownEstimandException(Exception):
    def __init__(self):
        super().__init__("Unknown Estimand provided")

class UnknownDGPVariableException(Exception):
    def __init__(self):
        super().__init__("Unknown DGP Variable provided")

class DGPVariableMissingException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class ParameterSpecificationException(Exception):
    pass

class ParameterMissingFromSpecException(ParameterSpecificationException):
    def __init__(self, missing_param):
        super().__init__(f"The spec is missing the value for the parameter {missing_param}")

class ParameterInvalidValueException(ParameterSpecificationException):
    def __init__(self, invalid_param, invalid_value):
        super().__init__(f"The spec contains an  invalid value for the parameter {invalid_param}. The supplied value was {invalid_value}.")

class CalculatedParameterException(ParameterSpecificationException):
    def __init__(self, invalid_param):
        super().__init__(f"The spec contains an  value for the parameter {invalid_param}. No value should be supplied for calculated params.")
