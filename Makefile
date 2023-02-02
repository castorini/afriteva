.PHONY: create-env

create-env:
		python3 -m pip install --upgrade pip
		python3 -m pip install --user -r requirements.txt
