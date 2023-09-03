from flask import Flask, jsonify, request
import convert
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/get_variable')


def get_variable():
    input_value = request.args.get('input')
    variable_value = convert.createTodoRequestLists(input_value)
    return jsonify(variable=variable_value)

if __name__ == '__main__':
    app.run()
