FROM yazansh/pytorch-lightning:latest


COPY requirements.txt additional-requirements.txt
RUN pip install -r additional-requirements.txt
