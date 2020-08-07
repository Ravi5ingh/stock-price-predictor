# Stock Price Predictor
This is the source code for a web app that is being built to make predictions about equities. This is very much a work in progress and the nightly build is running <a href="http://159.65.28.124:3002/">here<a>

## Project Definition

### Overview
This is the source code for a web app whose main aim is to display the results of models created to try and predict the price movements of stocks. The models were trained exclusively on equities data but they are agnostic to the asset class. All the data has been sourced from Yahoo Finance.

### Problem Statement
The problem that needs to be solved is quite straight forward. We want to be able to accurately predict where the price of the stock will be in the next few days so that it could be used for buy/sell recommendations. The strategy is to experiment with different models until a suitable one is found. These models are discussed in more detail in the 'models' section.

### Metrics
The data used is the stock price data from Yahoo Finance. The metric used to assess a model is the root mean square error (RMSE).

## Analysis & Conclusion
The full investigation process is detailed in the 'models' folder.

## Setup
This is a PyCharm project with a virtual python environment (the 'venv' folder is git ignored). To run this app locally, follow these instructions:
<ol>
    <li>Clone the source code in this repo</li>
    <li>Get PyCharm from <a href="https://www.jetbrains.com/pycharm/download">here</a> </li>
    <li>Launch PyCharm</li>
    <li>Create new project</li>
    <li>Select the folder that you cloned the source code to</li>
    <li>This project was built with Python 3.7.3 x64 (Virtual Env) so select python environment accordingly</li>
    <li>If you intend on using virtual environment, when PyCharm asks whether or not to create from existing sources? Say NO</li>
    <li>This will create a virtual environment for you</li>
    <li>In command prompt, navigate to the root directory and run the following commands:</li>
    <li>pip install pandas</li>
    <li>pip install yfinance</li>
    <li>pip install lxml</li>
    <li>pip install matplotlib</li>
    <li>pip install pyyaml</li>
    <li>pip install sklearn</li>
    <li>pip install keras</li>
    <li>pip install tensorflow</li>
    <li>pip install keras</li>
    <li>pip install numpy</li>
    <li>pip install keras</li>
    <li>pip install flask</li>
    <li>pip install json</li>
    <li>pip install plotly</li>
    <li>In the root dir, run 'python -m website.run'</li>
    <li>If there are no error messages, the website is running <a href="http://localhost:3001">here</a></li>
</ol>