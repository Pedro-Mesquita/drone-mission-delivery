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

## Como a missão voa

Foto a foto, sempre a 4 m de altura:

1. Decola até 4 m.
2. Para, espera o drone estabilizar e tira 3 fotos (`/take_photo`, `rpicam-still`, 1280×720).
3. Projeta no chão as cartolinas detectadas pelo `ObbDetector`; uma cartolina vira
   obstáculo quando aparece em 2 das 3 fotos. O chão coberto pela foto fica marcado como visto.
4. Planeja com A* até o alvo e voa só a parte da rota sobre chão já visto (com a margem
   de segurança em volta do drone). Volta ao passo 2.
5. No destino, abre o gripper (canal 9, PWM 1400) e volta ao HOME do mesmo jeito, onde pousa.

Pousa onde está se não houver caminho livre ou se uma foto falhar 5 vezes. Em qualquer
outro erro (inclusive Ctrl+C), tenta pousar e mostra o erro. Os valores (altura, número
de fotos, pista, tolerâncias) ficam em `src/mission_delivery/config.py`; a pista padrão
é provisória (30 × 10 m a partir do ponto de decolagem).

```bash
python -m mission_delivery.mission --target 28 0 --base-url http://127.0.0.1:8020
```

`--target` é `NORTE LESTE [ALTITUDE]` em metros a partir do HOME; a altitude é ignorada.
`--model` troca o arquivo do modelo (padrão `models/ground-obstacle-v2.pt`).

## Simulação (Gazebo + ArduPilot SITL)

`sim/` tem a pista de referência: cartolina à direita (norte 8 m), à esquerda (15 m) e
duas deixando uma só passagem (22 m); destino norte 28 m. O drone (`iris_downcam`) é o
iris do ArduPilot com uma câmera fixa para baixo com os parâmetros da Camera Module 3.

Na simulação, `sim/bin/rpicam-still` substitui a câmera: a UAV API roda com `sim/bin` no
`PATH` e o `/take_photo` devolve o quadro do Gazebo. O código da missão é o mesmo do drone.

```bash
UAV_API_PYTHON=/caminho/para/python-com-uav_api \
UAV_API_CONFIG=/caminho/para/config.ini \
sim/run_sim.sh            # GUI=1 abre a janela do Gazebo

# em outro terminal, depois de "uav api: starting"
python -m mission_delivery.mission --target 28 0 --base-url http://127.0.0.1:8020
```

Requisitos: ArduPilot (`~/ardupilot`), plugin `~/ardupilot_gazebo` compilado, MAVProxy em
`~/venv-ardupilot` com `gz` e OpenCV no mesmo Python. `sim/make_world.py` regenera o mundo.
