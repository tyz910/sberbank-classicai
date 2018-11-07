from flask import Flask, request, jsonify, abort, render_template
from util.poem import FinalPoemGenerator

app = Flask(__name__, template_folder="assets/templates", static_folder="assets/static")
gen = FinalPoemGenerator()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ready')
def ready():
    return 'OK'


@app.route('/generate/<poet_id>', methods=['POST'])
def generate(poet_id):
    request_data = request.get_json()
    seed = request_data['seed']
    try:
        generated_poem = gen.generate(poet_id, seed)
        return jsonify({'poem': generated_poem})
    except KeyError:
        abort(404)


if __name__ == '__main__':
    print(">>>> START SERVER. http://0.0.0.0:8000/")
    app.run(host='0.0.0.0', port=8000)
