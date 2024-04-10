# Stage 1: Build the frontend
FROM node:20 as build-stage
WORKDIR /app
COPY ./frontend/package*.json ./
RUN npm install
COPY ./frontend ./
RUN npm run build

# Stage 2: Run the backend
FROM python:3.11-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV PORT=8080
WORKDIR /app
COPY --from=build-stage /app/build ./react
COPY ./backend ./
RUN pip install -r requirements.txt
EXPOSE ${PORT}
CMD uvicorn deploy:app --host 0.0.0.0 --port ${PORT}