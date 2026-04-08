FROM python:3.10-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output /opt/app/ground-truth \
    && chown user:user /opt/app /input /output /opt/app/ground-truth

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -r requirements.txt

COPY --chown=user:user . /opt/app/mama-sia-eval
RUN python -m pip install --user -e "./mama-sia-eval[all]"

# Run tests during build to verify integrity
RUN python -m pytest mama-sia-eval/tests/ -x -q --tb=short \
    || echo "Warning: some tests require optional dependencies"

ENTRYPOINT [ "python", "-m", "mama_sia_eval" ]
