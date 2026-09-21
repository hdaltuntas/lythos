"""Upload a built Lythos release to PyPI, from an editor such as Thonny.

Press the green Run button and answer the questions in the Shell.  The
script does what `twine upload` does on a terminal, but it builds its own
environment to run it from, so nothing has to be installed system-wide -
which is what makes it work on Arch, where pip refuses to touch the system
Python.

What it does, in order:

    1. makes a virtual environment in ~/.lythos-release and puts twine in it
    2. finds the .whl and .tar.gz of one release
    3. runs `twine check` on them
    4. asks for your API token, unless ~/.pypirc already holds one
    5. asks you to confirm, then uploads

Nothing is sent anywhere before step 5, and answering anything but "yes"
stops it.

Remember that PyPI accepts a version number once and only once: read what
step 3 prints before you confirm.
"""

import configparser
import os
import re
import subprocess
import sys
import venv

# Set this to True to rehearse on TestPyPI, a throwaway copy of PyPI with its
# own accounts and its own tokens.  Nothing you upload there is permanent.
TEST_PYPI = False

ENV = os.path.expanduser("~/.lythos-release")
PYPIRC = os.path.expanduser("~/.pypirc")
REPOSITORY = "testpypi" if TEST_PYPI else "pypi"
TOKEN_PAGE = ("https://test.pypi.org/manage/account/token/" if TEST_PYPI
              else "https://pypi.org/manage/account/token/")

# Where a downloaded release is likely to be sitting.
SEARCH = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dist"),
    os.path.abspath("dist"),
    os.path.expanduser("~/lythos-dist"),
    os.path.expanduser("~/Downloads/dist"),
    os.path.expanduser("~/Downloads"),
    os.path.dirname(os.path.abspath(__file__)),
]

VERSION = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+?)-(?P<version>\d[^-]*?)"
                     r"(?:-py3-none-any)?\.(?:whl|tar\.gz)$")


def ask(question, default=None):
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{question}{suffix}: ").strip()
    except EOFError:
        return default or ""
    return answer or (default or "")


def yes(question):
    return ask(f"{question} (yes/no)", "no").lower() in ("y", "yes", "e", "evet")


# ------------------------------------------------------------------ the tools
def tool(name):
    """Path to a program inside our own environment."""
    binary = "Scripts" if os.name == "nt" else "bin"
    suffix = ".exe" if os.name == "nt" else ""
    return os.path.join(ENV, binary, name + suffix)


def prepare_environment():
    if not os.path.isfile(tool("python")):
        print(f"Making a small environment in {ENV} ...")
        venv.EnvBuilder(with_pip=True, clear=True).create(ENV)
    if not os.path.isfile(tool("twine")):
        print("Installing twine into it ...")
        run([tool("python"), "-m", "pip", "install", "--quiet",
             "--disable-pip-version-check", "--upgrade", "twine"],
            "twine could not be installed")
    print("Ready.\n")


def run(command, failure, env=None):
    result = subprocess.run(command, env=env)
    if result.returncode != 0:
        raise SystemExit(f"\n{failure} (exit code {result.returncode}).")


# --------------------------------------------------------------- the packages
def releases_in(folder):
    """Group the distribution files in one folder by version."""
    found = {}
    if not os.path.isdir(folder):
        return found
    for entry in sorted(os.listdir(folder)):
        match = VERSION.match(entry)
        if match:
            key = (match.group("name"), match.group("version"))
            found.setdefault(key, []).append(os.path.join(folder, entry))
    return found


def choose_files():
    for folder in SEARCH:
        folder = os.path.normpath(folder)
        found = releases_in(folder)
        if found:
            return pick(folder, found)

    print("No .whl or .tar.gz file was found in any of the usual places:")
    for folder in SEARCH:
        print("   ", os.path.normpath(folder))
    while True:
        folder = ask("\nWhere are they? (paste the folder, or press Enter to give up)")
        if not folder:
            raise SystemExit("Nothing to upload.")
        folder = os.path.expanduser(folder.strip().strip("'\""))
        found = releases_in(folder)
        if found:
            return pick(folder, found)
        print(f"Nothing that looks like a release in {folder}.")


def pick(folder, found):
    print(f"Found in {folder}:")
    keys = sorted(found)
    for index, (name, version) in enumerate(keys, start=1):
        files = ", ".join(os.path.basename(f) for f in found[(name, version)])
        print(f"  {index}. {name} {version} - {files}")

    if len(keys) == 1:
        key = keys[0]
    else:
        # More than one version in the folder: uploading the wrong one cannot
        # be undone, so never guess.
        print("\nThere is more than one release here.")
        while True:
            answer = ask("Which number")
            if answer.isdigit() and 1 <= int(answer) <= len(keys):
                key = keys[int(answer) - 1]
                break
            print("Type one of the numbers above.")

    files = found[key]
    kinds = {os.path.splitext(f)[1] for f in files}
    if ".whl" not in kinds:
        print("\nWarning: there is no .whl file, only the source archive.")
    if not any(f.endswith(".tar.gz") for f in files):
        print("\nWarning: there is no .tar.gz source archive, only the wheel.")
    return key, files


# ------------------------------------------------------------------ the token
def stored_token():
    """A token already in ~/.pypirc, if there is one for this repository."""
    if not os.path.isfile(PYPIRC):
        return None
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(PYPIRC)
    except configparser.Error:
        return None
    if parser.has_option(REPOSITORY, "password"):
        return parser.get(REPOSITORY, "password")
    return None


def get_token():
    token = stored_token()
    if token:
        print(f"Using the token already in {PYPIRC}.\n")
        return token, False

    print(f"An API token is needed.  Get one from {TOKEN_PAGE}")
    print("  - scope: 'Entire account' for a project's first upload")
    print("  - it starts with 'pypi-' and it is shown once only")
    print("\nWhat you type next WILL be visible in the Shell.\n")
    token = ask("Token")
    if not token:
        raise SystemExit("No token, no upload.")
    if not token.startswith("pypi-"):
        print("\nThat does not look like a token - they begin with 'pypi-'.")
        if not yes("Use it anyway?"):
            raise SystemExit("Stopped.")
    return token, True


def offer_to_save(token):
    print(f"\nThe token can be kept in {PYPIRC} so this is not asked again.")
    print("It is a password: the file is written readable by you alone.")
    if not yes("Save it?"):
        return
    parser = configparser.ConfigParser(interpolation=None)
    if os.path.isfile(PYPIRC):
        parser.read(PYPIRC)
    if not parser.has_section(REPOSITORY):
        parser.add_section(REPOSITORY)
    parser.set(REPOSITORY, "username", "__token__")
    parser.set(REPOSITORY, "password", token)
    # Create it with the right permissions from the start, rather than
    # writing the token first and narrowing afterwards.
    handle = os.open(PYPIRC, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(handle, "w", encoding="utf-8") as fh:
        parser.write(fh)
    os.chmod(PYPIRC, 0o600)
    print(f"Saved to {PYPIRC}.")


# ------------------------------------------------------------------- the deed
def main():
    where = "TestPyPI (a rehearsal - nothing published there is permanent)" \
        if TEST_PYPI else "PyPI (the real one)"
    print(f"Uploading to {where}.\n")

    prepare_environment()
    (name, version), files = choose_files()

    print("\nChecking the files ...")
    run([tool("twine"), "check", *files], "The files did not pass the check")

    print("\n" + "-" * 62)
    print(f"About to upload {name} {version}:")
    for path in files:
        print(f"    {os.path.basename(path)}  "
              f"({os.path.getsize(path) / 1024:.0f} kB)")
    print(f"  to {where}")
    print("\nA version number is accepted once and never again.  If anything")
    print("in it is wrong, stop here, fix it, raise the version and rebuild.")
    print("-" * 62 + "\n")
    if not yes("Upload now?"):
        raise SystemExit("Stopped.  Nothing was sent.")

    token, is_new = get_token()

    environment = dict(os.environ)
    environment["TWINE_USERNAME"] = "__token__"
    environment["TWINE_PASSWORD"] = token
    environment["TWINE_NON_INTERACTIVE"] = "1"
    if TEST_PYPI:
        environment["TWINE_REPOSITORY"] = "testpypi"

    print("\nUploading ...\n")
    run([tool("twine"), "upload", *files], "The upload failed",
        env=environment)

    site = "https://test.pypi.org" if TEST_PYPI else "https://pypi.org"
    print(f"\nDone.  {site}/project/{name}/{version}/")
    print(f"Anyone can now run:  pip install {name}")

    if is_new:
        offer_to_save(token)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as stop:
        print(stop)
    except KeyboardInterrupt:
        print("\nStopped.")
