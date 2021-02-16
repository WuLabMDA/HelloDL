# HelloDL
Run the PyTorch MNIST classification with Docker utilizing GPU

### Step 1: Clone the repo
```
$ cd $HOME
$ git clone https://github.com/WuLabMDA/HelloDL.git
$ cd HelloDL
```

### Step 2: Build docker image
* Start build docker image
```
$ docker build -t hello-dl .
```

### Step 3: Create docker container
* Start create docker container based on the built image
```
$ docker run -it --gpus all --rm --user $(id -u):$(id -g) \
    -v /home/pchen6/HelloDL:/home/pchen6/HelloDL \
    --name hello hello-dl:latest
```

### Step 4: Run MNIST classification
* (optional) Check GPU available or not
```
$ gpustat
```

* Train the PyTorch DL model
```
$ python main.py
```
* **99.54%** accuracy can be obtained for **mnist-classification** after training for 10 epochs.
