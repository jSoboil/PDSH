# Activating virtual env. -------------------------------------------------
# Load reticulate package:
library(reticulate)
# Load virtual env:
use_virtualenv("env./")
# Point reticulate to Python version of choice:
Sys.setenv(RETICULATE_PYTHON = "env./bin/python")
reticulate::repl_python()

# End file ----------------------------------------------------------------