import flask as fl
import utility.config as cf
import yfinance as yf
import json as js
import plotly as pl
from flask import request as rq

app = fl.Flask(__name__)

def __get_graph_for_symbol__(symbol, period):
    """
    Get the data mapping for the daily close prices for the given stock symbol for the time period
    :param symbol: The stock symbol
    :param period: The time period (eg. '1Y') in accordance with Yahoo Finance
    :return: The data and layout mapping
    """
    stock = yf.Ticker(symbol).history(period=period)

    return [
        dict(
            data=[
                dict(
                    x=stock.index,
                    y=stock['Close'],
                    type='line'
                )
            ],
            layout=dict(
                title=f'Daily closing price for {symbol}'
            )
        )
    ]

# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Try and get the stock symbol and default to Microsoft
    stock_symbol = rq.args.get('symbol')
    stock_symbol = stock_symbol if stock_symbol is not None else 'MSFT'

    # Get the daily closing prices for the given stock (default to 1 year for now)
    graphs = __get_graph_for_symbol__(stock_symbol, period='1Y')

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = js.dumps(graphs, cls=pl.utils.PlotlyJSONEncoder)

    return fl.render_template('master.html', ids=ids, graphJSON=graphJSON)

def main():
    app.run(host='0.0.0.0', port=3002, debug=cf.is_app_debug())

if __name__ == '__main__':
    main()