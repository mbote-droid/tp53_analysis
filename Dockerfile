# ============================================================
# TP53 Bioinformatics Analysis Pipeline — Docker Image
# ============================================================
# Base image: official Python 3.10 slim (smaller than full)
FROM python:3.10-slim

# Metadata
LABEL maintainer="Samuel Mbote"
LABEL description="TP53 Bioinformatics Analysis Pipeline"
LABEL version="1.0.0"

# Set working directory inside container
WORKDIR /app

# Install system dependencies needed by matplotlib and BioPython
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker caches this layer —
# only rebuilds if requirements.txt changes, not your code)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Create results directory
RUN mkdir -p results

# Set matplotlib to non-interactive backend
ENV MPLBACKEND=Agg

# Default environment variable placeholder
# Override at runtime: docker run -e ENTREZ_EMAIL=your@email.com
ENV ENTREZ_EMAIL=""

# Expose Streamlit port
EXPOSE 8501

# Default command runs the Streamlit web app
# To run the CLI instead:
# docker run -e ENTREZ_EMAIL=your@email.com tp53-pipeline python main_tp53_analysis.py --accession NM_000546
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]