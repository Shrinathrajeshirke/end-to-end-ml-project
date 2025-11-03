FROM python:3.10-slim-buster
WORKDIR /app

# copy requirements
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# copy application code
COPY . .   

# expose port
EXPOSE 5000

# use gunicorn
CMD ["gunicorn", "--bind", '0.0.0.0:5000', "--workers", "2", "--timeout", "120", "app:app" ]