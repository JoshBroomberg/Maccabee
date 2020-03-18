# Modeling Exceptions

class UnknownEstimandException(Exception):
    def __init__(self):
        super().__init__("Unknown Estimand provided")

class UnknownEstimandAggregationException(Exception):
    def __init__(self):
        super().__init__("Estimand with unknown aggregation provided")

# DGP exceptions

class UnknownDGPVariableException(Exception):
    def __init__(self):
        super().__init__("Unknown DGP Variable provided")

class DGPVariableMissingException(Exception):
    def __init__(self, msg):
        super().__init__(msg)

class DGPInvalidSpecificationException(Exception):
    def __init__(self, method_obj):
        super().__init__(f"Invalid DGP class specification. {method_obj} is a _generate* method without the data_generating_method decorator.")

class DGPFunctionCompilationException(Exception):
    def __init__(self, base_exception):
        super().__init__(f"Failure in compilation of expression. Root exception: {e}")


# Parameter Exceptions

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
        super().__init__(f"The spec contains a concrete value for the parameter {invalid_param} but the parameter is of the calculated type. No value should be supplied for calculated parameters.")
