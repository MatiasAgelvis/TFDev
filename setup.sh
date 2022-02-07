
# check if graphics card drivers are installed
if command -v nvidia-smi &> /dev/null
then
    # pip install tensorflow-gpu
    CONTAINER='tensorflow/tensorflow:2.7.0-gpu-jupyter'
  else
    # pip install tensorflow
    CONTAINER='tensorflow/tensorflow:2.7.0-jupyter'
fi

docker pull $CONTAINER
# pip install -r requirements.txt
docker run -p8888:8888 $CONTAINER
