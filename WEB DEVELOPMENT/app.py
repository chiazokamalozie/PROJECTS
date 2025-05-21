from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index')  # ✅ handles both / and /index
def index():
    return render_template('index.html')

@app.route('/services')
def services():
    return render_template("Services.html")

@app.route('/programs')
def programs():
    return render_template("Programs.html")

@app.route('/donate')
def donate():
    return render_template("Donate.html")

@app.route('/ourblog')
def ourblog():
    return render_template('ourblog.html')

@app.route('/about')
def about():
    return render_template("About.html")

@app.route('/team')
def team():
    return render_template("team.html")

if __name__ == '__main__':
    app.run(debug=True)
