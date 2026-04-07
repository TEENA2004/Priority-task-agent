FROM python:3.11-slim

# Working directory set karo
WORKDIR /app

# Requirements install karo
RUN pip install openenv-core pydantic fastapi uvicorn

# Apna code copy karo
COPY priority_task_env.py .
COPY openenv.yaml .

# Port expose karo
EXPOSE 8000

# Environment start karo
CMD ["uvicorn", "priority_task_env:app", "--host", "0.0.0.0", "--port", "8000"]
