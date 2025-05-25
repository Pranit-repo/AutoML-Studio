import threading
import subprocess
from flask import Flask, render_template, redirect, render_template_string, url_for


apps = [
    ("data_c/clean_main.py", 5003, "Data Cleaning Dashboard"),
    ("EDA/eda_main.py", 5002, "Exploratory Data Analysis"),
    ("feature_e/feature_main.py", 5004, "Feature Engineering"),
    ("model/model_main.py", 5001, "Model Training & Evaluation"),
]


def run_app(file_name, port):
    subprocess.Popen(["python", file_name])


app = Flask(__name__)


DASHBOARD_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Data Science Workflow Dashboard</title>
  <style>
    :root {
      --primary: #6c8fc7;
      --secondary: #4a6fa5;
      --accent: #4fc3f7;
      --background: #121212;
      --card-bg: #1e1e1e;
      --text: #e0e0e0;
      --text-secondary: #b0b0b0;
      --border: #333;
      --card-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      margin: 0;
      padding: 0;
      background-color: var(--background);
      color: var(--text);
      min-height: 100vh;
    }
    
    .container {
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem;
    }
    
    header {
      text-align: center;
      margin-bottom: 2.5rem;
      padding-bottom: 1rem;
      border-bottom: 1px solid var(--border);
    }
    
    h1 {
      color: var(--primary);
      margin-bottom: 0.5rem;
      font-weight: 600;
    }
    
    .subtitle {
      color: var(--text-secondary);
      font-size: 1.1rem;
    }
    
    .apps-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
      gap: 1.5rem;
    }
    
    .app-card {
      background: var(--card-bg);
      border-radius: 8px;
      overflow: hidden;
      box-shadow: var(--card-shadow);
      transition: transform 0.2s ease, box-shadow 0.2s ease;
      border: 1px solid var(--border);
      display: flex;
      flex-direction: column;
    }
    
    .app-card:hover {
      transform: translateY(-5px);
      box-shadow: 0 10px 20px rgba(0, 0, 0, 0.4);
      border-color: var(--primary);
    }
    
    .app-card-content {
      padding: 1.5rem;
      flex: 1;
      display: flex;
      flex-direction: column;
    }
    
    .app-card h3 {
      margin-top: 0;
      color: var(--primary);
      font-size: 1.25rem;
    }
    
    .app-card p {
      color: var(--text-secondary);
      margin-bottom: 1.5rem;
    }
    
    .app-link-container {
      margin-top: auto;
      padding-top: 1rem;
    }
    
    .app-link {
      display: block;
      background-color: var(--primary);
      color: white;
      padding: 0.75rem 1.5rem;
      border-radius: 4px;
      text-decoration: none;
      font-weight: 500;
      transition: all 0.2s ease;
      text-align: center;
      width: 100%;
      box-sizing: border-box;
    }
    
    .app-link:hover {
      background-color: var(--secondary);
      transform: translateY(-2px);
    }
    
    footer {
      text-align: center;
      margin-top: 3rem;
      padding-top: 1.5rem;
      border-top: 1px solid var(--border);
      color: var(--text-secondary);
      font-size: 0.9rem;
    }
    
    @media (max-width: 768px) {
      .apps-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Machine Learning Workflow Dashboard by AutoML AI Studio</h1>
      <p class="subtitle">Access different stages of the Machine Learning pipeline</p>
    </header>
    
    <div class="apps-grid">
      {% for app in apps %}
        <div class="app-card">
          <div class="app-card-content">
            <h3>{{ app.name }}</h3>
            <p>Access the {{ app.name }} module to perform specific tasks in the workflow.</p>
            <div class="app-link-container">
              <a href="{{ app.url }}" target="_blank" class="app-link">Open</a>
            </div>
          </div>
        </div>
      {% endfor %}
    </div>
    
    <footer>
      <p>© 2025 Machine Learning Workflow. All rights reserved by AutoML AI Studio.</p>
    </footer>
  </div>
</body>
</html>
"""


@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    app_links = [{"name": name, "url": f"http://127.0.0.1:{port}/"} for _, port, name in apps]
    return render_template_string(DASHBOARD_TEMPLATE, apps=app_links)

def run_flask_app():
    app.run(port=8000)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)

    for file_name, port, _ in apps:
        threading.Thread(target=run_app, args=(file_name, port)).start()
    run_flask_app()
