# syntax=docker/dockerfile:1
FROM python:3.13-slim AS build
WORKDIR /src
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir build && python -m build --wheel --outdir /dist

FROM python:3.13-slim
RUN useradd --system --uid 10001 --home /app memsvc && mkdir -p /app /models && chown memsvc /models
COPY --from=build /dist /dist
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "$(ls /dist/*.whl)[embed]" && rm -rf /dist
ENV HF_HOME=/models PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
# Bake the embedding model into the image: no download at start, no outbound access needed.
RUN python -c "from transformers import AutoModel, AutoTokenizer as T; n='sentence-transformers/all-MiniLM-L6-v2'; T.from_pretrained(n); AutoModel.from_pretrained(n)" \
 && chown -R memsvc /models
ENV HF_HUB_OFFLINE=1
USER 10001
WORKDIR /app
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8080/healthz').status != 200)"
ENTRYPOINT ["memsvc"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080"]
