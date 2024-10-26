from flask import Flask
from flask import send_from_directory
from flask import render_template
from pathlib import Path

app = Flask(__name__)

def load_results():
    results = {}
    base_path = Path("../results/")

    for dataset_path in base_path.iterdir():
        if dataset_path.is_dir():
            dataset_name = dataset_path.name
            results[dataset_name] = {}
            
            for run_path in dataset_path.iterdir():
                if run_path.is_dir():
                    stats_file = run_path / "stats.txt"
                    results_file = run_path / "results.txt"
                    
                    if stats_file.is_file():
                        with stats_file.open('r') as f:
                            stats = {}
                            for line in f:
                                key, value = line.strip().split(": ", 1)
                                stats[key] = value
                            results[dataset_name][run_path.name] = {"stats": stats}
                    
                    if results_file.is_file():
                        with results_file.open('r') as f:
                            results_data = {}
                            for line in f:
                                key, value = line.strip().split(": ", 1)
                                values_list = value.split(", ")
                                if key == "Discovered concept semantics":
                                    results_data[key] = values_list
                                else:
                                    results_data[key] = [float(v) if v.lower() != "nan" else float('nan') for v in values_list]
                            results[dataset_name][run_path.name]["results"] = results_data

    
    return results

results = load_results()

@app.route("/")
def main_page():
    return render_template("index.html", results=results, selected_dataset=None)

@app.route('/<dataset>/<run>')
def run_page(dataset, run):
    if dataset in results and run in results[dataset]:
        run_details = results[dataset][run]
        print(run_details)
        return render_template('run.html', results=results, selected_dataset=dataset, run=run, run_details=run_details)
    else:
        abort(404)

# @app.route('/datasets/<path:path>')
# def serve_datasets(path):
#     return send_from_directory('/datasets', path)
