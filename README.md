Install Ollama

`brew install ollama or curl -fsSL https://ollama.com/install.sh | sh`

Start the server

`ollama serve`

Clone the repo git@github.com:prabhakarjuzgar/local_llm.git

Create venv

`python -m venv .env`

`source .env/bin/activate`

`pip install -r requirements.txt`

Execute docker-compose.yml to have mongodb listening on 27018

`docker compose up -d`

Execute/Run fastapi app

`fastapi run main.py`

This will produce at endpoint similar to http://0.0.0.0:8000

Append docs and paste it in a browser - http://0.0.0.0:8000/docs

-----------------------------------------------------------------

Use app without fastapi
Excecute following on the cli
`cd into the root folder of the repo`

`python src/chat/local_llm_mongo.py <file_path> <query>`

Make sure mongodb is running - `docker compose up -d`
