import flask as fl
import utility.config as cf

app = fl.Flask(__name__)

# index web page displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    return fl.render_template('master.html')

def main():
    app.run(host='0.0.0.0', port=3002, debug=cf.is_app_debug())

if __name__ == '__main__':
    main()