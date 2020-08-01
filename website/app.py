import flask as fl
import utility.config as cf
import json as js
import plotly as pl
from flask import request as rq
import website.visualizer as vz

app = fl.Flask(__name__)

# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # Try and get the stock symbol and default to Microsoft
    stock_symbol = rq.args.get('symbol')
    stock_symbol = stock_symbol if stock_symbol is not None else 'MSFT'

    # Get the daily closing prices for the given stock (default to 1 year for now)
    graphs = [
        vz.get_stock_ts(stock_symbol, period='1Y'),
        vz.get_last_day_predictions(stock_symbol, period='1mo')
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = js.dumps(graphs, cls=pl.utils.PlotlyJSONEncoder)

    return fl.render_template('master.html', ids=ids, graphJSON=graphJSON)

def main():
    app.run(host='0.0.0.0', port=3002, debug=cf.is_app_debug())

if __name__ == '__main__':
    main()