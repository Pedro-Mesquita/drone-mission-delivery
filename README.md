# Mission Delivery

Missão autônoma: decolar, navegar até uma coordenada desviando de obstáculos
no chão (câmera apontada para baixo) e abrir o gripper no destino.

Este repositório contém só a lógica específica da missão. Comandos de voo e
log vêm de `drone-mission-core`; detecção em imagem vem de `drone-vision-core`.

## Instalação

```bash
pip install -e ".[dev]"        # roda o modelo .onnx (onnxruntime)
pip install -e ".[dev,pt]"     # também o .pt (ultralytics + torch), para exportar
```

## Modelo

O modelo YOLO-OBB não fica no git; o `.pt` é publicado como Release.

```bash
curl -L -o models/ground-obstacle-v2.pt \
  https://github.com/Pedro-Mesquita/drone-mission-delivery/releases/download/ground-obstacle-v2/ground-obstacle-v2.pt
```

SHA-256: `82abc1966ece5d4b5ad917101ee2eb0296feda982bb62523c5b0d73fd13d08c2`

A missão usa o `.onnx` (roda só com `onnxruntime`, sem torch), publicado na mesma Release:

```bash
curl -L -o models/ground-obstacle-v2.onnx \
  https://github.com/Pedro-Mesquita/drone-mission-delivery/releases/download/ground-obstacle-v2/ground-obstacle-v2.onnx
```

SHA-256: `7f9f5f56662c7244012ccfd44f4981c0fd35734b27a31344e86eb9244f05fe34`

Ele foi exportado do `.pt` com entrada fixa 640×384, a proporção das fotos
1280×720 (precisa do extra `pt`):

```bash
python -c 'from ultralytics import YOLO; YOLO("models/ground-obstacle-v2.pt").export(format="onnx", imgsz=[384, 640], simplify=True)'
```

Nas fotos 1280×720 ele acha as mesmas caixas que o `.pt`.

## Raspberry Pi

O Pi não precisa de internet nem de acesso aos repositórios da equipe. No PC,
com o `.onnx` em `models/`:

```bash
pi/pack.sh 3.11      # versão do Python do Pi (python3 --version); Bookworm = 3.11
```

Isso baixa para `pi/wheels/` tudo o que a missão usa, compilado para ARM 64 bits.
Copie o repositório para o Pi e rode lá:

```bash
rsync -a --exclude .venv --exclude flight_logs --exclude sim ./ pi@<ip-do-pi>:drone-mission-delivery/
ssh pi@<ip-do-pi> drone-mission-delivery/pi/setup.sh
```

(`scp -r` também serve, mas copia o `.venv` e os logs do PC à toa.) O `setup.sh`
cria o `.venv`, instala tudo de `pi/wheels/` e roda o modelo uma vez para
conferir. Precisa de Raspberry Pi OS 64 bits e do mesmo Python usado no
`pack.sh`. A UAV API não é instalada por ele.

## Como a missão voa

Foto a foto, sempre a 4 m de altura:

1. Decola até 4 m.
2. Para, espera o drone estabilizar e tira 3 fotos (`/take_photo`, `rpicam-still`, 1280×720).
3. Projeta no chão as cartolinas detectadas pelo `ObbDetector`; uma cartolina vira
   obstáculo quando aparece em 2 das 3 fotos. O chão coberto pela foto fica marcado como visto.
4. Planeja com A* até o alvo e voa só a parte da rota sobre chão já visto (com a margem
   de segurança em volta do drone). Volta ao passo 2.
5. No destino, abre o gripper (canal 9, PWM 1400) e volta ao HOME do mesmo jeito, foto a foto,
   com o mapa recomeçando vazio (o que foi visto na ida não conta). No HOME, pousa.

Pousa onde está se não houver caminho livre ou se uma foto falhar 5 vezes. Em qualquer
outro erro (inclusive Ctrl+C), tenta pousar e mostra o erro. Os valores (altura, número
de fotos, pista, tolerâncias) ficam em `src/mission_delivery/config.py`; a pista padrão
é provisória (30 × 10 m a partir do ponto de decolagem).

```bash
python -m mission_delivery.mission --target 28 0 --base-url http://127.0.0.1:8020
```

`--target` é `NORTE LESTE [ALTITUDE]` em metros a partir do HOME; a altitude é ignorada.
`--model` troca o arquivo do modelo (padrão `models/ground-obstacle-v2.onnx`; um `.pt` também serve).

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
