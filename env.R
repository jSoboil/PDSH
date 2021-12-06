library(reticulate)
# create a new environment 
# virtualenv_create("pdsd_venv/")
Sys.setenv(RETICULATE_PYTHON = "pdsd_venv/bin/python")
# install SciPy
# virtualenv_install("numpy")

# import SciPy (it will be automatically discovered in "r-reticulate")
# scipy <- import("scipy")