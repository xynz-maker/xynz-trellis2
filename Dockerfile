# Xynz 3D Studio — servidor de IA 3D (TRELLIS.2 da Microsoft, licença MIT).
# Atualização rápida: parte da imagem completa já montada (receita em Dockerfile.full) e só troca o handler.
FROM ghcr.io/xynz-maker/xynz-trellis2:3ebb1b61c31761cfc7c96ef7e6dea4f38cc8f9c5
COPY diag.py /app/diag.py
RUN cd /app/TRELLIS.2 && python /app/diag.py 2>&1 | tail -120; true
COPY handler.py /app/handler.py
WORKDIR /app/TRELLIS.2
CMD ["python", "-u", "/app/handler.py"]
