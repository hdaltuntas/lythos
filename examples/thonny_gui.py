"""Start the Lythos interface from Thonny.

Press the green Run button.  A browser tab opens with the interface in it;
load an example from the top of the page, or draw a section yourself.

The script keeps running while the interface is open - that is the server
doing its job.  Press Thonny's red Stop button when you are done.

    pip install lythosfea
"""

import webbrowser

PORT = 8777


def main() -> None:
    try:
        from lythos.gui.server import serve
    except ImportError:
        print("Lythos is not installed in the interpreter Thonny is using.")
        print("In Thonny: Tools > Manage packages… > search 'lythosfea' > Install")
        return

    url = f"http://127.0.0.1:{PORT}/"
    print("Lythos is starting.")
    print(f"If the browser does not open by itself, go to {url}")
    print("Press the red Stop button in Thonny to shut it down.\n")

    # Thonny sometimes runs without a usable browser hook, so open it here and
    # let the server itself skip that step.
    try:
        webbrowser.open(url)
    except Exception:
        pass
    serve(port=PORT, open_browser=False)


if __name__ == "__main__":
    main()
