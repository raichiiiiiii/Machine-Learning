from pathlib import Path
from time import perf_counter
from threading import Thread, Event
import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

BASE_DIR = Path(__file__).resolve().parent

NOTEBOOKS = [
    "notebook/preprocess.ipynb",
    "notebook/wrapper.ipynb",
    "notebook/PCA.ipynb",
    "notebook/RNN.ipynb",
    "notebook/MLP.ipynb",
    "notebook/RNN_analysis.ipynb",
    "notebook/MLP_analysis.ipynb",
]


def fmt_time(seconds):
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def print_cell_outputs(cell):
    for output in cell.get("outputs", []):
        output_type = output.get("output_type")

        if output_type == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            print(text, end="", flush=True)

        elif output_type == "error":
            for line in output.get("traceback", []):
                print(line, flush=True)

        elif output_type in ("execute_result", "display_data"):
            data = output.get("data", {})
            text = data.get("text/plain")
            if text:
                if isinstance(text, list):
                    text = "".join(text)
                print(text, flush=True)


def live_timer(label, start_time, stop_event):
    while not stop_event.is_set():
        elapsed = fmt_time(perf_counter() - start_time)
        msg = f"\r{label} | elapsed {elapsed}"
        print(msg.ljust(120), end="", flush=True)
        stop_event.wait(0.2)  # refresh 5 times/sec

    elapsed = fmt_time(perf_counter() - start_time)
    msg = f"\r{label} | elapsed {elapsed}"
    print(msg.ljust(120), end="", flush=True)


overall_start = perf_counter()

for nb_name in NOTEBOOKS:
    nb_path = BASE_DIR / nb_name
    nb_start = perf_counter()

    print(f"\n{'=' * 90}")
    print(f"RUNNING: {nb_path.name}")
    print(f"{'=' * 90}")

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    code_cells = [
        (i, cell)
        for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and cell.source.strip()
    ]

    total_code_cells = len(code_cells)
    client = NotebookClient(nb, timeout=None)

    try:
        with client.setup_kernel(cwd=str(nb_path.parent)):
            for step, (i, cell) in enumerate(code_cells, start=1):
                cell_start = perf_counter()
                label = f"[{step}/{total_code_cells}] {nb_path.name} | cell {i + 1}"

                stop_event = Event()
                timer_thread = Thread(
                    target=live_timer,
                    args=(label, cell_start, stop_event),
                    daemon=True
                )
                timer_thread.start()

                try:
                    client.execute_cell(cell, i)
                finally:
                    stop_event.set()
                    timer_thread.join()

                # move to next line after live counter
                print()
                print_cell_outputs(cell)

                print(
                    f"[done] {label} | cell took {fmt_time(perf_counter() - cell_start)}",
                    flush=True
                )

    except CellExecutionError:
        print(f"\nFAILED: {nb_path.name}", flush=True)
        with open(nb_path, "w", encoding="utf-8") as f:
            nbformat.write(nb, f)
        raise

    with open(nb_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(
        f"FINISHED: {nb_path.name} | notebook took {fmt_time(perf_counter() - nb_start)} | "
        f"total elapsed {fmt_time(perf_counter() - overall_start)}",
        flush=True
    )

print(f"\nAll notebooks finished in {fmt_time(perf_counter() - overall_start)}")