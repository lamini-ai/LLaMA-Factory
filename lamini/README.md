# Lamini fork of Llama Factory (https://github.com/hiyouga/LLaMA-Factory)

This is for added features needed for Lamini Platform.

## Naming

For branch names, we use the following naming convention:
`lamini-<LLAMA_FACTORY_VERSION>`

e.g. `lamini-v0.9.1`, `lamini-v0.9.0`, etc.

For image tags, we use the following naming convention:
`<LLAMA_FACTORY_VERSION>-<LAMINI_VERSION>`. Please create a release (https://github.com/lamini-ai/LLaMA-Factory/releases) to match new image tags.

e.g. `v0.9.1-000`, `v0.9.1-001`, `v0.9.1-002`, etc.

## Updating from upstream
On the latest lamini branch, checkout one for the new version (this will keep all lamini commits)

```bash
# Create a new branch, e.g. 'lamini-v0.9.1'
$ git checkout -b lamini-<LLAMA_FACTORY_VERSION>

# Fetch upstream tags
$ git fetch upstream --tags

# Rebase
$ git rebase <LLAMA_FACTORY_VERSION>

# Push
git push --set-upstream origin lamini-<LLAMA_FACTORY_VERSION>
```

Finally, update the default branch in the repo settings to the latest version.
<img width="852" alt="Screenshot 2025-02-07 at 6 44 44 PM" src="https://github.com/user-attachments/assets/031cc358-b501-4f98-afd0-9d3801e9a5be" />

## Github Actions

Github Actions have been disabled for this repo to avoid triggering vLLM's workflows. They can be re-enabled in the repo settings.

## Building the Docker image

### RoCM

```bash
docker build -f ./docker/docker-rocm/Dockerfile \
    --build-arg INSTALL_BNB=false \
    --build-arg INSTALL_VLLM=false \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg INSTALL_FLASHATTN=false \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --platform linux/amd64 \
    -t ghcr.io/lamini-ai/lamini-tuning-engine-amd:<INSERT_TAG> .
```

<details>
<summary>Push the image to GHCR</summary>
If you want to push the image to GHCR, you can use the following commands:

```bash
docker push ghcr.io/lamini-ai/lamini-tuning-engine-amd:<INSERT_TAG>
```

</details>

### Nvidia

```bash
docker build -f ./docker/docker-cuda/Dockerfile \
    --build-arg INSTALL_BNB=false \
    --build-arg INSTALL_VLLM=false \
    --build-arg INSTALL_DEEPSPEED=false \
    --build-arg INSTALL_FLASHATTN=false \
    --build-arg PIP_INDEX=https://pypi.org/simple \
    --platform linux/amd64 \
    -t ghcr.io/lamini-ai/lamini-tuning-engine-nvidia:<INSERT_TAG> .
```

<details>
<summary>Push the image to GHCR</summary>
If you want to push the image to GHCR, you can use the following commands:

```bash
docker push ghcr.io/lamini-ai/lamini-tuning-engine-nvidia:<INSERT_TAG>
```

</details>
