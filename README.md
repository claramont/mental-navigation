# Mental Navigation

Python project by Clara Montemurro and Francesca Dessalvi to reproduce the figures of the paper [Mental navigation in the primate entorhinal cortex](https://www.nature.com/articles/s41586-024-07557-z#Sec22), by Neupane e co.
The code is organized with `src/` and the environment is handled through [Poetry](https://python-poetry.org/) to manage dependencies.


## Struttura del progetto

```
mental-navigation/
  src/mental_navigation/
      __init__.py
      Fig1
        ... (data and scripts to reproduce fig 1)
      Fig2
        ... (data and scripts to reproduce fig 1)
      ...
  tests/
    ... (test)
  pyproject.toml
```

## Project setup
### 1. Verify Python version / Install
Verify if you have Python already installed:

**macOS / Linux:**
```bash
python3 --version
```
**Windows (Powershell o CMD)**:
```bash
python --version
```
Verify , it satisfies the requirements:  `>=3.11`.


### 2. Install Poetry (if not already done)
Verify if you have Poetry installed
```bash
poetry --version
```
To install Poetry, follow the instructions either using [pipx](https://python-poetry.org/docs/#installing-with-pipx) or using the [official installer](https://python-poetry.org/docs/#installing-with-the-official-installer)

### 3. Clone the repo with HTTPS URL:
Navigate to the local folder where you want to clone the repo;

Run: ```bash git clone https://github.com/claramont/mental-navigation.git ```

Navigate to the local project repository del progetto:
```bash
cd mental-navigation
```

### 4. Create the virtualenv and install dependencies
In the terminal (we used Visual Studio Code, but any IDE works as well), run:
```bash
poetry install
```
This will install the packages, handle the dependencies needed and create the virtualenv.

To activate the environment, run: 
```bash
poetry shell
```
**Poetry 2.x** does not include `poetry shell` bu default, so depending on the poetry version you istalled, you might need to:
- Use new command (reccommended)
  ```bash
  poetry env activate```
- Install the plugin to have poetry shell:
  ```bash
  poetry self add poetry-plugin-shell
  ```
  then run:
  ```bash
  poetry shell
  ```
- Select the desired interpreter using Visual Studio Code.
  Then, every built-in terminal will be open inside the venv, without the need to run ```poetry shell```


