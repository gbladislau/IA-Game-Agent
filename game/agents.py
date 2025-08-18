import numpy as np
from abc import ABC, abstractmethod
from typing import List

from game.config import GameConfig

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
    def __init__(self, config:GameConfig, weights:np.ndarray, n_neuronios_l1=32, n_neuronios_l2=16, n_neuronios_out=3):
        self.config = config
        
        tamanho_entrada = config.sensor_grid_size * config.sensor_grid_size + 2
        self.neuronios_layer_1 = n_neuronios_l1
        self.weights_layer_1 = weights[
            :tamanho_entrada * self.neuronios_layer_1].reshape(
            n_neuronios_l1, tamanho_entrada)
        size_entrada = tamanho_entrada * self.neuronios_layer_1
        
        self.neuronios_layer_2 = n_neuronios_l2
        self.weights_layer_2 = weights[
            size_entrada:size_entrada+n_neuronios_l2*n_neuronios_l1].reshape(
            n_neuronios_l2,n_neuronios_l1)
        size_l1 = n_neuronios_l2*n_neuronios_l1
        
        self.neuronios_layer_out = n_neuronios_out
        self.weights_layer_3 = weights[
            size_entrada+size_l1 : size_entrada+size_l1 + n_neuronios_out*n_neuronios_l2].reshape(
            n_neuronios_out,n_neuronios_l2)
        size_l2 = n_neuronios_out*n_neuronios_l2
        
        self.bias_layer_1 = weights[size_entrada+size_l1+size_l2: size_entrada+size_l1+size_l2 + n_neuronios_l1].reshape(n_neuronios_l1,1)
        self.bias_layer_2 = weights[size_entrada+size_l1+size_l2+ n_neuronios_l1 : size_entrada+size_l1+size_l2 + n_neuronios_l1 + n_neuronios_l2 ].reshape(n_neuronios_l2,1)
        self.bias_layer_3 = weights[size_entrada+size_l1+size_l2+n_neuronios_l1+ n_neuronios_l2 :].reshape(n_neuronios_out,1)
        
        self.n_total_param_treinaveis = tamanho_entrada * n_neuronios_l1 + n_neuronios_l1 * n_neuronios_l2 + n_neuronios_l2 * n_neuronios_out +  n_neuronios_l1 + n_neuronios_l2 + n_neuronios_out
    
    def get_n_total_trainable_params(self) -> int:
        return self.n_total_param_treinaveis
    
    @staticmethod
    def softmax(x: np.ndarray):
        return np.exp(x)/sum(np.exp(x))
    
    def predict(self, state: np.ndarray) -> int:
        entrada = state.flatten()
        
        camada1_lin_out = np.dot(self.weights_layer_1, entrada) + self.bias_layer_1
        camada1_atv_out = np.tanh(camada1_lin_out)
        
        camada2_lin_out = np.dot(self.weights_layer_2, camada1_atv_out) + self.bias_layer_2
        camada2_atv_out = np.tanh(camada2_lin_out)
        
        camada3_lin_out = np.dot(self.weights_layer_3, camada2_atv_out) + self.bias_layer_3
        camada3_atv_out = self.softmax(camada3_lin_out)
        # print(camada3_atv_out)
        return int(np.argmax(camada3_atv_out))