# ---------- FinSmart AI FastAPI Dockerfile ----------

# 1️⃣ Base image: lightweight Python
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy all project files into container
COPY . .

# 4️⃣ Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 5️⃣ Expose the port Hugging Face expects (7860)
EXPOSE 7860

# 6️⃣ Start the FastAPI app using uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

