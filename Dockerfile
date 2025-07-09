FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["jupyter", "nbconvert", "--to", "notebook", "--execute", "Ice_Classification.ipynb", "--output", "executed.ipynb"]