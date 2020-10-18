# HelloDL
Run the PyTorch MNIST classification with Docker utilizing GPU

### Step 1: Clone the repo
```
$ git clone https://github.com/WuLabMDA/HelloDL.git
$ cd HelloDL
```

### Step 2: Build docker image
```
$ docker build -t hello-dl .
```

### Step 3: Create docker container
```
$ docker run -it --gpus all --rm \
    --user $(id -u):$(id -g) \
    -v /home/pchen6/Codes/HelloDL:/home/pchen6/HelloDL \
    --name hello hello-dl:latest
```

### Step 4: Run MNIST classification
* First check whether NVIDIA GPU exist via
```
$ nvidia-smi # or
$ gpustat
```
* Enter to the code folder
```
$ cd HelloDL
```
* Train the PyTorch DL model
```
$ python main.py
```
