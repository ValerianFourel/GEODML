# CPU image for the full run. The code is NOT baked in: run with the pinned clean checkout mounted at /code and
# GEODML_GIT_COMMIT set to its commit, so git stays the source of truth and the image only changes with dependencies.
#   docker build -t geodml-cpu:1 analysis/container
#   docker run --rm -v "$PWD":/code -e GEODML_GIT_COMMIT=$(git rev-parse HEAD) geodml-cpu:1 python -m pytest -q analysis/tests/test_fullrun_ledger.py
# HoreKa: docker save geodml-cpu:1 -o geodml-cpu.tar; copy; apptainer build geodml-cpu.sif docker-archive://geodml-cpu.tar
FROM public.ecr.aws/docker/library/python:3.12-slim
COPY requirements-cpu.lock /tmp/requirements-cpu.lock
RUN pip install --no-cache-dir -r /tmp/requirements-cpu.lock
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/code
WORKDIR /code
LABEL org.opencontainers.image.title="geodml-cpu" org.opencontainers.image.description="CPU analysis stages of the GEODML full run"
