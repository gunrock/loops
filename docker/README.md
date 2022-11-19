# Docker Instructions
## Using the provided `dockerfile`, simply run the following commands.
```bash
docker build . -t loops
docker run -it loops:latest /bin/bash
```

## Alternatively pull directly from [hub.docker.com](https://hub.docker.com/repository/docker/neoblizz/loops).

```bash
docker pull neoblizz/loops:v0.1
```

## Once within the docker...
```bash
cd loops/build
bin/loops.spmv.merge_path -m ../datasets/chesapeake/chesapeake.mtx
```