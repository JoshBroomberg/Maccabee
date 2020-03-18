import logging

# The code below does not configure the root logger or add handlers
# to the loggers used in the submodules. This allows client/user code to
# specify and control logging output.
# This pattern is as suggesting the Python logging documentation
# https://docs.python.org/3/howto/logging.html

LOG_LEVEL = logging.DEBUG

# Build a logger with the given name and level.
def get_logger(logger_name, log_level=LOG_LEVEL):
   logger = logging.getLogger(logger_name)
   logger.setLevel(log_level)

   return logger

def get_package_root_logger():
    return(get_logger("maccabee"))
