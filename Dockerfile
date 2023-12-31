# This is a potassium-standard dockerfile, compatible with Banana

# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Fix bitsandbytes to compile with gpu support
RUN cp /opt/conda/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cuda113.so /opt/conda/lib/python3.8/site-packages/bitsandbytes/libbitsandbytes_cpu.so

# Workaround
# Cannot import name 'Self' from 'typing_extensions'
# https://github.com/Azure/azure-sdk-for-python/issues/28651
RUN pip3 install typing_extensions==4.4.0

# Add your model weight files 
# (in this case we have a python script)
ADD download.py .
RUN python3 download.py

ADD . .

EXPOSE 8000

CMD python3 -u app.py