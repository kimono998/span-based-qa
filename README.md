# Span-based Question Answering

## Project Overview

This project involves fine-tuning a Huggingface transformer model to perform span-based question answering. Given a context and a question, the model is trained to predict the start and end tokens of the answer from the context.

## Course

This project was completed as part of an **Explainable AI** course.

## Task

The goal is to fine-tune a pre-trained transformer model to accurately predict answer spans from the input context and question.

## Models Used and Parameters

- **Model**: `DistilBertForQuestionAnswering`
    - A lightweight version of BERT used for question-answering tasks.

- **Tokenizer**: `DistilBertTokenizerFast`
    - Efficiently encodes the input context and question pairs into token sequences suitable for the model.

### Hyperparameters

The following hyperparameters were used for fine-tuning:

1. **Batch Size**: 8
2. **Learning Rate**: 5e-5
3. **Optimizer**: AdamW
