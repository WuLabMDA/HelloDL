# HelloDL
Run the PyTorch MNIST classification with Docker utilizing GPU

### Step 1: Clone the repo
```
$ git clone https://github.com/WuLabMDA/HelloDL.git
$ cd HelloDL
```

### Step 2: Build docker image
* Update **USER_NAME** in line 12 of Dockerfile to your name
```
$ docker build -t hello-dl .
```

### Step 3: Create docker container
* Go back to parent folder
```
$ cd ..
```
* Start docker container
```
$ docker run -it --gpus all --rm \
    --user $(id -u):$(id -g) \
    --name hello hello-dl:latest
```

### Step 4: Run MNIST classification
* First check whether NVIDIA GPU exist or not
```
$ gpustat
```
* Train the PyTorch DL model
```
$ python main.py
```
