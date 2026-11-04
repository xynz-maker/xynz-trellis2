# Xynz 3D Studio — servidor de IA 3D (TRELLIS.2 da Microsoft, licença MIT).
# Atualização rápida: parte da imagem completa já montada (receita em Dockerfile.full, CUDA 12.8) e só troca o handler.
FROM ghcr.io/xynz-maker/xynz-trellis2:9f566130efac0ce0f5ce0446e3cf486d7aa51c31
COPY handler.py /app/handler.py
WORKDIR /app/TRELLIS.2
CMD ["python", "-u", "/app/handler.py"]
