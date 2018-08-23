#load Flask
import flask
app = flask.Flask(__name__)

# Define a predict function as an endpoint
@app.route("/prediction", methods=["GET", "POST"])

def predict():
    data = {"success": False}

    # Get request parameters
    params = flask.request.json
    if (params == None):
        params = flask.request.args

    # If parameters are found, echo the msg parameter
    if (params != None):
        data["response"] = params.get("msg")
        data["success"] = True

    # Return a response in json format
    return flask.jsonify(data)

# Start the flask app
app.run()
