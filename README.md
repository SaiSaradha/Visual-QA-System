# Visual-QA-System

A simple Visual Question Answering (VQA) system was built and the performance of two different neural
network models - Baseline Multi Layer Perceptron (MLP) and Long Short Term Memory (LSTM) -
were compared with respect to their accuracy and loss. VQA is an interesting and complex AI task
involving multiple disciplines such as Computer Vision, Natural Language Processing and Machine
Learning where a machine is trained to answer questions about an image shown to it. In this
project, I have used a deep learning approach to setup the VQA system by training the neural
networks on the VQA V2.0 open-ended dataset . Training was implemented on Google Cloud VM
instances and evaluation of the trained model was performed locally. Training both the models for
just 50 epochs resulted in promising results with accuracy values reaching 25.34% for MLP and
22.15% for LSTM, however, preliminary observations in extended training showed indications that
LSTM outperforms MLP (data not shown).
