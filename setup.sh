# check if graphics card drivers are installed
if command -v nvidia-smi &> /dev/null
then
    pip install tensorflow-gpu
else
    pip install tensorflow
fi

pip install -r requirements.txt