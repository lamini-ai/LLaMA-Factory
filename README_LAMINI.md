# Lamini Fork of LLaMAFactory

```shell
VERSION=v[Version]
cd docker/docker-cuda
docker compose build
docker tag docker-cuda-llamafactory ghcr.io/lamini-ai/lamini-tuning-engine-nvidia:${VERSION}

export CR_PAT=YOUR_TOKEN
echo $CR_PAT | docker login ghcr.io -u USERNAME --password-stdin
docker push ghcr.io/lamini-ai/lamini-tuning-engine-nvidia:${VERSION}
```
