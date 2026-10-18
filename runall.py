"""ONE command for everything:  python runall.py

1. runs the full pipeline (data -> clean -> EDA -> model -> insights)
2. starts the themed web dashboard + REST API on http://127.0.0.1:5000

Options:
  --no-server       run the pipeline only
  --skip-pipeline   reuse the saved model and just start the API
  --port 8000       use another port
"""
import argparse
import threading
import webbrowser

from src import config as C


def main():
    parser = argparse.ArgumentParser(description="Employee Performance Predictor")
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    if not args.skip_pipeline or not C.MODEL_PATH.exists():
        from src.pipeline import run_pipeline
        run_pipeline()

    if args.no_server:
        return

    from src.api import create_app
    url = f"http://127.0.0.1:{args.port}"
    print(f"\n== Starting API + dashboard at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    create_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
