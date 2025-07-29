import numpy as np
from abc import ABC, abstractmethod
from typing import List

class Agent(ABC):
    """Interface para todos os agentes."""
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        """Faz uma previsão de ação com base no estado atual."""
        pass

class HumanAgent(Agent):
    """Agente controlado por um humano (para modo manual)"""
    def predict(self, state: np.ndarray) -> int:
        # O estado é ignorado - entrada vem do teclado
        return 0  # Padrão: não fazer nada (será sobrescrito pela entrada do usuário no manual_play.py)

class NeuralNetworkAgent(Agent):
    def __init__(self):
        
        pass
    
    def predict(self, state: np.ndarray) -> int:
        
        grid_flat = state[:self.grid_size*self.grid_size]
        grid = grid_flat.reshape((self.grid_size, self.grid_size))
        player_y_normalized = state[-2] * self.config.screen_height # Second last element
        center_row = self.grid_size // 2
        
        pass
