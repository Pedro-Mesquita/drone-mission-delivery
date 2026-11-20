# Mission Delivery

Missão autônoma: decolar, navegar até uma coordenada desviando de obstáculos
no chão (câmera apontada para baixo) e abrir o gripper no destino.

Este repositório contém só a lógica específica da missão. Comandos de voo e
log vêm de `drone-mission-core`; detecção em imagem vem de `drone-vision-core`.

## Instalação

```bash
pip install -e ".[dev]"
```

## Modelo

O modelo YOLO-OBB não fica no git. Baixe o arquivo da Release correspondente
e coloque em `models/` (ex.: `models/ground-obstacle-v2.pt`).
