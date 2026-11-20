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

O modelo YOLO-OBB não fica no git; é publicado como Release.

```bash
curl -L -o models/ground-obstacle-v2.pt \
  https://github.com/Pedro-Mesquita/drone-mission-delivery/releases/download/ground-obstacle-v2/ground-obstacle-v2.pt
```

SHA-256: `82abc1966ece5d4b5ad917101ee2eb0296feda982bb62523c5b0d73fd13d08c2`
