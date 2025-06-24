FROM python:3.10

RUN useradd -m user

WORKDIR /home/user/project
COPY ./requirements.txt .

RUN pip3 install -r requirements.txt

USER user
COPY --chown=user:user . .

EXPOSE 8080

CMD ["python3", "main.py"]

