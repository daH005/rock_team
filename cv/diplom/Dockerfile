FROM python:3.10

RUN useradd -m user

WORKDIR /home/user/project
COPY ./requirements.txt .

# Dependencies for CV2:
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

# Python libraries:
RUN pip3 install -r requirements.txt

USER user
COPY --chown=user:user . .

EXPOSE 8080

WORKDIR /home/user
CMD ["python3", "-m", "project.main"]

