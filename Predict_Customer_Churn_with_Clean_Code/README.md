# Predict Customer Churn


## Project Overview
The objective of this project is predict the customer churn implementing engineering and software best practices.

## Installation
To run this project you will need to have Python 3.x installed on your machine

You can run this project using Docker
## Pre requisites
    - [Docker](https://docs.docker.com/install/) `2.1.0.1` or greater.

Ensuring that you have Docker

```bash
$ docker -v

Docker version 20.10.17, build 100c701
```

Build the image

```bash
$ docker build --tag prediction-churn .
```
Check the image with the following command
```bash
$ docker images
```

## Run prediction pipeline
```bash
$ docker run -t prediction-churn python churn_library.py   
```

## Run prediction test pipeline
```bash
$ docker run -t prediction-churn pytest churn_script_logging_and_test.py   
```