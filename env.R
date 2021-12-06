# Load reticulate package:
library(reticulate)

# create a new environment/select venv:
# virtualenv_create("pdsd_venv/")
Sys.setenv(RETICULATE_PYTHON = "pdsd_venv/bin/python")

# Install numpy to venv:
# virtualenv_install("numpy")