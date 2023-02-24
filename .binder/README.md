# Binder files

This folder contains:
-  `environment.yml`: file needed to configure a Docker image on Binder with the requisite packages. It is simply a link to `osca.yml` in the main repository folder. 
- `requirements.txt`: file listing the python package requirements for the repository, which is read by the environment file. It is also simply a link to `requirements.txt` in the main repository folder.
- `apt.txt`: a list of additional libraries to be installed in the Docker image for the [opencv-python](https://exerror.com/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directory/) package.
