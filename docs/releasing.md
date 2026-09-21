# Releasing to PyPI

The distribution is named `lythosfea`; the package you import and the command
you type stay `lythos`.

## Without a terminal

`tools/upload_to_pypi.py` does all of the below from an editor: open it in
Thonny (or IDLE, or VS Code), press Run, and answer the questions in the
shell pane.  It builds its own environment for twine, so the system Python is
left alone - on Arch, where pip refuses to install into it, that is the
difference between working and not.  Set `TEST_PYPI = True` at the top of the
file to rehearse.

Nothing is sent before it has printed what it is about to upload and you have
answered `yes`.

## Once, before the first upload

Get an API token from <https://pypi.org/manage/account/token/>.  Until the
project exists on PyPI the token has to be account-wide; afterwards replace it
with one scoped to `lythosfea`.  Put it in `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...
```

`chmod 600 ~/.pypirc`.  The token is a password - it never belongs in the
repository.

## Every release

1. Bump `version` in `pyproject.toml`.  **PyPI accepts a version number once
   and only once**, so anything wrong in the metadata - the author name, the
   licence, the README that becomes the project page - has to be fixed before
   the upload, not after.
2. Run the tests: `pytest -q`.
3. Build clean:

   ```bash
   rm -rf dist build
   python -m build
   twine check dist/*
   ```

4. Upload:

   ```bash
   twine upload dist/*
   ```

5. Check what was actually published, in an empty environment:

   ```bash
   python -m venv /tmp/check
   /tmp/check/bin/pip install lythosfea
   /tmp/check/bin/lythos examples -o /tmp/check-models
   /tmp/check/bin/lythos run /tmp/check-models/slope.json -o /tmp/check-out
   ```

6. Tag it: `git tag v0.1.0 && git push --tags`.

## Rehearsing on TestPyPI

TestPyPI is a separate site with its own account and its own token
(<https://test.pypi.org/manage/account/token/>; add it to `~/.pypirc` under a
`[testpypi]` section):

```bash
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ lythosfea
```

The second index is needed because NumPy, SciPy and Matplotlib are not
mirrored there.  A name used on TestPyPI does not reserve it on PyPI.

## What goes into the wheel

`lythos/gui/static/*` and `lythos/data/*.dxf` are declared as package data in
`pyproject.toml`.  Without them the interface serves a blank page and
`lythos import --sample` has nothing to read, and neither failure shows up in
the tests, which run from the source tree.  After building, look:

```bash
python -c "import zipfile; print(zipfile.ZipFile('dist/lythosfea-0.1.0-py3-none-any.whl').namelist())"
```
