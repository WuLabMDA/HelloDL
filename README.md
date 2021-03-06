# HelloDL
Run the PyTorch MNIST classification with Docker utilizing GPU

### Step 1: Clone the repo
```
$ cd ${HOME}
$ git clone https://github.com/WuLabMDA/HelloDL.git
$ cd HelloDL
```

### Step 2: Build docker image
* Start build docker image
```
$ docker build -t hellodl .
```

### Step 3: Create docker container
* Start create docker container based on the built image
```
$ DOCKER_CODE_DIR=/App/HelloDL
$ docker run -it --gpus all --rm --user $(id -u):$(id -g) \
    -v ${PWD}:${DOCKER_CODE_DIR} --name hello hellodl:latest
```

### Step 4: Run MNIST classification
* (optional) Check GPU available or not
```
$ gpustat
```

* Train the PyTorch DL model
```
$ DOCKER_CODE_DIR=/App/HelloDL
$ cd ${DOCKER_CODE_DIR}
$ python main.py
```
* **99.50%** accuracy can be obtained for **mnist-classification** after training for 10 epochs.
