# Mental Navigation

Progetto Python per riprodurre figure del paper di Neupane e co (link paper).  
Il codice è organizzato con layout `src/` e gestito tramite [Poetry](https://python-poetry.org/) per la gestione delle dipendenze.


## Struttura del progetto

```
mental-navigation/
  src/
    mental_navigation/
      __init__.py
      ... (codice applicativo)
  tests/
    ... (test)
  pyproject.toml
```

## Setup progetto
### 1. Verifica / installa versione Python
Verifica se Python è già installato e assicurati che la versione sia >=3.11:
**macOS / Linux:**

```bash
python3 --version
```
**Windows (Powershell o CMD)**:
```bash
python --version
```

### 2. Installare Poetry (se non è già presente)
Verifica se Poetry è installato:
```bash
poetry --version
```
Se non è presente, seguire le info al link https://python-poetry.org/docs/#installing-with-the-official-installer

### 3. Clonare la repo con URL HTTPS:
Navigare nella cartella locale dove si vuole clonare la repo
Eseguire git clone https://github.com/claramont/mental-navigation.git nel terminale
Navigare nella repo locale del progetto: cd mental-navigation;

### 4. Creare l'ambiente e installare le dipendenze
Eseguire poetry install nel terminale di VSCode per installare le dipendenze
Per attivare l'ambiente, eseguire poetry shell (o poetry env activate)

