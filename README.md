# Portefólio de Aprendizagem por Reforço (RL)

**Mestrado em Informática e Inteligência Artificial (MIA)**  
**Aprendizagem por Reforço — 2.º Semestre 2025/2026**

---

Este repositório contém o pacote Python unificado, modular e de nível de produção **`mia_rl`**, desenvolvido no âmbito da unidade curricular de Aprendizagem por Reforço. O pacote consolida todos os exercícios práticos da disciplina (desde bandits multi-braço a gradientes de política e Monte Carlo Tree Search) sob um framework orientado a objetos comum.

Adicionalmente, inclui uma pasta **experimental** para exploração de algoritmos de controlo avançados (Double Q-learning e DQN em PyTorch) com suporte a monitorização gráfica em tempo real via **TensorBoard**.

---

## 📂 Estrutura do Repositório

O projeto está organizado da seguinte forma:

```
portefolio/
│
├── mia_rl/                     # Pacote Python principal
│   ├── core/                   # Interfaces genéricas (Environment, Agent, Policy, etc.)
│   ├── envs/                   # Ambientes customizados (Blackjack, Windy Gridworld, TicTacToe)
│   ├── mdps/                   # Modelos MDP para Programação Dinâmica
│   ├── agents/                 # Algoritmos de RL padrão (Prediction, Control, MCTS)
│   ├── policies/               # Políticas reutilizáveis de seleção de ações
│   ├── features/               # Extratores de características (Tile coding, One-hot)
│   ├── experiments/            # Treino, rollouts de trajetórias e torneios
│   ├── scripts/                # Scripts utilitários executáveis por linha de comandos
│   ├── notebooks/              # Jupyter Notebooks de suporte às aulas
│   ├── experimental/           # Secção de exploração avançada (Double Q-learning, PyTorch DQN)
│   └── outputs/                # Gráficos e logs do TensorBoard gerados em runtime
│
├── _test_portfolio.py          # Conjunto de testes de fumo automatizados (smoke tests)
└── environment.yml             # Definição do ambiente Conda
```

---

## 🛠️ Filosofia de Design e Engenharia de Software

O portefólio adota boas práticas de engenharia de software aplicadas à Inteligência Artificial:
1.  **Modularidade estrita:** Separação clara entre as regras do ambiente, o modelo de decisão do agente (pesos/redes neuronais), o ciclo de recolha de episódios e a extração de características do estado.
2.  **Abstrações Partilhadas:** Todos os agentes herdam de classes abstratas base (`Agent`, `PredictionAgent`, `ControlAgent`). Isto permite que o mesmo ciclo de treino corra indistintamente com um agente tabular clássico ou com uma rede neuronal profunda.
3.  **Monitorização Moderna:** Integração de barras de progresso diagnósticas (`tqdm`) e envio de métricas de treino (perdas, retornos, taxas de vitória) em tempo real para o **TensorBoard**.

---

## 🚀 Configuração do Ambiente

Inicialize o ambiente virtual Python utilizando a definição Conda fornecida:

```bash
# 1. Criar o ambiente a partir do ficheiro yml
conda env create -f environment.yml

# 2. Ativar o ambiente
conda activate rl

# 3. Validar a instalação correndo os testes automatizados
python _test_portfolio.py
```

---

## 🏃 Execução de Experiências (Scripts)

Todos os algoritmos podem ser corridos como scripts. As imagens geradas são guardadas em `mia_rl/outputs/`.

### 1. Benchmark Consolidado do Portefólio
Executa uma avaliação comparativa automática de vários agentes e imprime uma tabela sumária (MC vs TD(0) no Blackjack, variantes SARSA no Windy Gridworld, e REINFORCE vs MCTS no TicTacToe):
```bash
python -m mia_rl.scripts.run_benchmark
```

### 2. Prática 1: Bandits Multi-Braço
Compara estratégias de exploração vs. explotação ($\epsilon$-greedy, UCB e Gradient Bandits):
```bash
python -m mia_rl.scripts.run_kbandits
```

### 3. Prática 3: Blackjack (Previsão)
Estima a função de valor utilizando First-Visit Monte Carlo e Diferenças Temporais TD(0):
```bash
python -m mia_rl.scripts.run_blackjack_prediction
```

### 4. Prática 4: Windy Gridworld (Controlo Tabular)
Treina agentes SARSA tabular e SARSA $n$-passos:
```bash
# Executar SARSA Tabular
python -m mia_rl.scripts.run_windy_gridworld_sarsa

# Executar n-step SARSA
python -m mia_rl.scripts.run_windy_gridworld_n_step_sarsa
```

### 5. Prática 5: Aproximação de Funções (Semi-Gradient SARSA)
Utiliza aproximação linear (com tile-coding) e uma rede linear em PyTorch no Windy Gridworld:
```bash
# Linear TD(0)
python -m mia_rl.scripts.run_windy_gridworld_linear_td

# Linear SARSA Control (NumPy)
python -m mia_rl.scripts.run_windy_gridworld_linear_sarsa

# Torch SARSA Control (com seleção automática de CPU ou CUDA GPU)
python -m mia_rl.scripts.run_windy_gridworld_torch_sarsa --device auto
```

### 6. Práticas 6, 7 & 9: TicTacToe (REINFORCE e MCTS)
Treina e avalia agentes stocásticos e de planeamento no jogo do Galo:
```bash
# Treinar agente REINFORCE via self-play (gradiente de política)
python -m mia_rl.scripts.run_reinforce_tictactoe --episodes 15000

# Avaliar agente MCTS (Monte Carlo Tree Search) contra oponente aleatório
python -m mia_rl.scripts.run_mcts_tictactoe --simulations 300 --games 100
```

---

## 🔬 Secção Experimental: Explorações Adicionais

Como extensão ao programa curricular básico, implementámos dois algoritmos adicionais no módulo `experimental/` para avaliar problemas de viés de estimação e aproximadores não-lineares profundos:

### 1. Double Q-Learning Tabular
*   **Ficheiro:** `mia_rl/experimental/double_q.py`
*   **Conceito:** Mantém duas tabelas de valores-Q independentes ($Q_A$ e $Q_B$) para remover o viés de maximização inerente ao Q-learning clássico.

### 2. Deep Q-Network (DQN)
*   **Ficheiro:** `mia_rl/experimental/dqn.py`
*   **Conceito:** Implementação em PyTorch com **Experience Replay Buffer** e **Target Network** periódica para treinar um agente a jogar TicTacToe diretamente a partir do vetor de características do tabuleiro.

### Executar Treino Experimental e Visualizar no TensorBoard:
```bash
# 1. Correr o script de treino experimental (gera logs do TensorBoard)
python -m mia_rl.experimental.run_experimental --double-q-episodes 200 --dqn-episodes 1000

# 2. Iniciar o servidor TensorBoard para visualizar as curvas em tempo real
tensorboard --logdir mia_rl/outputs/runs/
```
Abra o endereço `http://localhost:6006/` no seu browser para acompanhar as perdas de treino do DQN e as curvas de recompensa do Double Q-learning.

---

## 📊 Tabela de Práticas e Algoritmos Implementados

| Prática | Tópico / Algoritmo | Ficheiro do Agente | Runner Executável |
| :--- | :--- | :--- | :--- |
| **P1** | Bandits Multi-Braço ($\epsilon$-greedy, UCB, Gradient Bandits) | `mia_rl/scripts/run_kbandits.py` | `python -m mia_rl.scripts.run_kbandits` |
| **P2** | Programação Dinâmica (Policy/Value Iteration) | `mia_rl/mdps/base.py` | `mia_rl/notebooks/MDP_GridWorld.ipynb` |
| **P3** | Previsão Baseada em Amostras (MC, TD(0)) | `mia_rl/agents/prediction/` | `python -m mia_rl.scripts.run_blackjack_prediction` |
| **P4** | Controlo Tabular (SARSA, $n$-step SARSA) | `mia_rl/agents/control/sarsa.py` | `python -m mia_rl.scripts.run_windy_gridworld_sarsa` |
| **P5** | Controlo com Função Linear/Torch (Semi-Gradient SARSA) | `mia_rl/agents/control/torch_sarsa.py` | `python -m mia_rl.scripts.run_windy_gridworld_torch_sarsa` |
| **P6** | Modelação de Ambientes Multijogador (TicTacToe) | `mia_rl/envs/tictactoe.py` | `mia_rl/notebooks/TicTacToe_Demo.ipynb` |
| **P7** | Gradientes de Política (REINFORCE em self-play) | `mia_rl/agents/control/reinforce.py` | `python -m mia_rl.scripts.run_reinforce_tictactoe` |
| **P9** | Planeamento em Tempo de Decisão (MCTS) | `mia_rl/agents/planning/mcts.py` | `python -m mia_rl.scripts.run_mcts_tictactoe` |
| **Exp** | Controlo Avançado (Double Q-learning, PyTorch DQN) | `mia_rl/experimental/` | `python -m mia_rl.experimental.run_experimental` |
