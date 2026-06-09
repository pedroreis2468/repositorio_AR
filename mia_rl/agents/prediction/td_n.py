from __future__ import annotations
from collections import defaultdict

from mia_rl.core.base import Episode, PredictionAgent
from mia_rl.envs.blackjack import BlackjackAction, BlackjackState

class NStepTDPrediction(PredictionAgent[BlackjackState, BlackjackAction]):
    def __init__(self, n: int, alpha: float = 0.05, gamma: float = 1.0):
        if n < 1:
            raise ValueError("O valor de 'n' deve ser pelo menos 1.")
        self.n = n
        self.alpha = alpha
        super().__init__(gamma=gamma)

    def reset(self) -> None:
        self.V = defaultdict(float) # Estimativa de valor do estado

    def update_episode(self, episode: Episode[BlackjackState, BlackjackAction]) -> None:
        """Atualiza a tabela de valores utilizando predição TD de n-passos."""
        T = len(episode.transitions)
        
        for t in range(T):
            state = episode.transitions[t].state
            
            # 1. Determinar o horizonte da janela de n-passos
            target_t = min(t + self.n, T)
            
            # 2. Calcular o retorno parcial truncado (soma das recompensas com desconto)
            G = 0.0
            for i in range(t, target_t):
                reward = episode.transitions[i].reward
                # O expoente do desconto é a distância do passo atual 't'
                G += (self.gamma ** (i - t)) * reward
            
            # 3. Adicionar o valor de bootstrap se a janela não atingiu o estado terminal
            if target_t < T:
                next_state = episode.transitions[target_t].state
                G += (self.gamma ** self.n) * self.V[next_state]
                
            # 4. Aplicar a atualização incremental TD
            self.V[state] += self.alpha * (G - self.V[state])

    def value_of(self, state: BlackjackState) -> float:
        return float(self.V[state])
