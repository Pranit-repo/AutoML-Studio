from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import zipfile
import io
import json
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SAVED_MODELS'] = 'saved_models'
app.config['STATIC_FILES'] = 'static_export'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SAVED_MODELS'], exist_ok=True)
os.makedirs(app.config['STATIC_FILES'], exist_ok=True)

MODELS = {
    'classification': {
        'Random Forest': {
            'class': RandomForestClassifier,
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'Logistic Regression': {
            'class': LogisticRegression,
            'params': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
        },
        'Gradient Boosting': {
            'class': GradientBoostingClassifier,
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        },
        'SVM': {
            'class': SVC,
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        'K-Nearest Neighbors': {
            'class': KNeighborsClassifier,
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
    },
    'regression': {
        'Random Forest': {
            'class': RandomForestRegressor,
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        },
        'Linear Regression': {
            'class': LinearRegression,
            'params': {}
        },
        'Ridge Regression': {
            'class': Ridge,
            'params': {
                'alpha': [0.1, 1, 10]
            }
        },
        'Lasso Regression': {
            'class': Lasso,
            'params': {
                'alpha': [0.1, 1, 10]
            }
        },
        'Gradient Boosting': {
            'class': GradientBoostingRegressor,
            'params': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        },
        'SVM': {
            'class': SVR,
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        'K-Nearest Neighbors': {
            'class': KNeighborsRegressor,
            'params': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            }
        }
    }
}

@app.route('/')
def index():
    return render_template('model_index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv')
        file.save(filepath)
        
        try:
            df = pd.read_csv(filepath)
            df = df.dropna(how='all')
            df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
            df.to_csv(filepath, index=False)
            
            columns = list(df.columns)
            preview = df.head().to_html(classes='table table-striped', index=False)
            
            target_col = columns[-1]
            unique_values = df[target_col].nunique()
            
            if df[target_col].dtype == 'object' or (unique_values < 10 and unique_values < len(df)/10):
                problem_type = 'classification'
            else:
                problem_type = 'regression'
                
            column_types = {}
            for col in columns:
                if df[col].dtype == 'object':
                    column_types[col] = 'categorical'
                else:
                    column_types[col] = 'numerical'
            
            return jsonify({
                'columns': columns,
                'preview': preview,
                'problem_type': problem_type,
                'column_types': column_types,
                'message': 'File uploaded successfully'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_model():
    data = request.json
    target = data['target']
    features = data['features']
    model_name = data['model']
    problem_type = data['problem_type']
    use_grid_search = data.get('use_grid_search', False)
    preprocess = data.get('preprocess', True)
    
    try:
        df = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'current_dataset.csv'))
        X = df[features]
        y = df[target]
        
        if preprocess:
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if X[col].nunique() > 10:
                    X = X.drop(col, axis=1)
                else:
                    X = pd.get_dummies(X, columns=[col], drop_first=True)
            
            if problem_type == 'classification' and y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                joblib.dump(le, os.path.join(app.config['SAVED_MODELS'], 'label_encoder.joblib'))
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model_config = MODELS[problem_type][model_name]
        model_class = model_config['class']
        model_params = model_config['params']
        
        if preprocess:
            pipeline_steps = [
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('model', model_class())
            ]
        else:
            pipeline_steps = [('model', model_class())]
        
        pipeline = Pipeline(pipeline_steps)
        
        if use_grid_search and model_params:
            grid_search = GridSearchCV(
                pipeline,
                {'model__' + key: value for key, value in model_params.items()},
                cv=5,
                scoring='accuracy' if problem_type == 'classification' else 'r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            model = grid_search.best_estimator_
            best_params = grid_search.best_params_
        else:
            model = pipeline
            model.fit(X_train, y_train)
            best_params = {}
        
        y_pred = model.predict(X_test)
        if problem_type == 'classification':
            score = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            metric_name = 'Accuracy'
            secondary_metric = 'F1 Score'
            secondary_score = f1
        else:
            score = mean_squared_error(y_test, y_pred, squared=False)
            r2 = r2_score(y_test, y_pred)
            metric_name = 'RMSE'
            secondary_metric = 'R2 Score'
            secondary_score = r2
        
        model_path = os.path.join(app.config['SAVED_MODELS'], 'current_model.joblib')
        joblib.dump(model, model_path)
        
        feature_names = list(X.columns)
        with open(os.path.join(app.config['SAVED_MODELS'], 'feature_names.json'), 'w') as f:
            json.dump(feature_names, f)
        
        return jsonify({
            'success': True,
            'score': score,
            'metric_name': metric_name,
            'secondary_metric': secondary_metric,
            'secondary_score': secondary_score,
            'model_name': model_name,
            'problem_type': problem_type,
            'best_params': best_params,
            'feature_names': feature_names
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['GET'])
def export_model():
    try:
        model_path = os.path.join(app.config['SAVED_MODELS'], 'current_model.joblib')
        if not os.path.exists(model_path):
            return jsonify({'error': 'No trained model found'}), 400
        
        metadata_path = os.path.join(app.config['SAVED_MODELS'], 'metadata.json')
        feature_names_path = os.path.join(app.config['SAVED_MODELS'], 'feature_names.json')
        
        metadata = {
            "platform": "No-Code AI Platform",
            "export_date": datetime.now().isoformat(),
            "requirements": ["scikit-learn", "pandas", "flask"]
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            zf.write(model_path, 'model/model.joblib')
            zf.write(metadata_path, 'model/metadata.json')
            
            if os.path.exists(feature_names_path):
                zf.write(feature_names_path, 'model/feature_names.json')
            
            label_encoder_path = os.path.join(app.config['SAVED_MODELS'], 'label_encoder.joblib')
            if os.path.exists(label_encoder_path):
                zf.write(label_encoder_path, 'model/label_encoder.joblib')
        
        memory_file.seek(0)
        
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='ai_model_package.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/deploy', methods=['POST'])
def deploy_model():
    try:
        deployment_code = "from flask import Flask, request, jsonify\n" + \
                         "import joblib\n" + \
                         "import pandas as pd\n" + \
                         "import numpy as np\n" + \
                         "from pathlib import Path\n\n" + \
                         "app = Flask(__name__)\n\n" + \
                         "model = joblib.load(Path('model') / 'model.joblib')\n\n" + \
                         "@app.route('/predict', methods=['POST'])\n" + \
                         "def predict():\n" + \
                         "    try:\n" + \
                         "        data = request.json\n" + \
                         "        features = data['features']\n" + \
                         "        input_data = pd.DataFrame([features], columns=model.feature_names_in_)\n" + \
                         "        prediction = model.predict(input_data)\n" + \
                         "        return jsonify({'prediction': prediction[0].tolist() if hasattr(prediction[0], 'tolist') else str(prediction[0])})\n" + \
                         "    except Exception as e:\n" + \
                         "        return jsonify({'error': str(e)}), 500\n\n" + \
                         "@app.route('/')\n" + \
                         "def home():\n" + \
                         "    return 'Model API is running. Use /predict endpoint with POST method.'\n\n" + \
                         "if __name__ == '__main__':\n" + \
                         "    app.run(host='0.0.0.0', port=5000)"
        
        memory_file = io.BytesIO()
        with zipfile.ZipFile(memory_file, 'w') as zf:
            model_path = os.path.join(app.config['SAVED_MODELS'], 'current_model.joblib')
            zf.write(model_path, 'model/model.joblib')
            
            feature_names_path = os.path.join(app.config['SAVED_MODELS'], 'feature_names.json')
            zf.write(feature_names_path, 'model/feature_names.json')
            
            label_encoder_path = os.path.join(app.config['SAVED_MODELS'], 'label_encoder.joblib')
            if os.path.exists(label_encoder_path):
                zf.write(label_encoder_path, 'model/label_encoder.joblib')
            
            zf.writestr('requirements.txt', "flask\nscikit-learn\npandas\nnumpy")
            zf.writestr('app.py', deployment_code)
        
        memory_file.seek(0)
        
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='model_deployment_package.zip'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_path = os.path.join(app.config['SAVED_MODELS'], 'current_model.joblib')
        model = joblib.load(model_path)
        
        with open(os.path.join(app.config['SAVED_MODELS'], 'feature_names.json')) as f:
            feature_names = json.load(f)
        
        data = request.json
        features = data['features']
        
        input_data = pd.DataFrame([features], columns=feature_names)
        prediction = model.predict(input_data)
        
        label_encoder_path = os.path.join(app.config['SAVED_MODELS'], 'label_encoder.joblib')
        if os.path.exists(label_encoder_path):
            le = joblib.load(label_encoder_path)
            prediction = le.inverse_transform(prediction)
        
        return jsonify({
            'prediction': prediction[0].tolist() if hasattr(prediction[0], 'tolist') else str(prediction[0])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)