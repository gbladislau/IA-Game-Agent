import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import NeuralNetworkAgent
from bat_algo import BAT_Algorithm
from network_test import test_agent
import os
import sys
import time
import multiprocessing as mp

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

POPULATION_SIZE = 100
GENERATIONS = 1000
RENDER = False
FPS = 60

def run_single_game(agents, game_config):
    game = SurvivalGame(config=game_config, render=RENDER)
    local_scores = np.zeros(len(agents))

    while not game.all_players_dead():
        actions = []
        for idx, agent in enumerate(agents):
            if game.players[idx].alive:
                state = game.get_state(idx, include_internals=True)
                action = agent.predict(state)
                actions.append(action)
            else:
                actions.append(0)
        game.update(actions)
        if game.render:
            game.render_frame()

    for idx, player in enumerate(game.players):
        local_scores[idx] += player.score

    return local_scores

def game_fitness_function(population: np.ndarray) -> np.ndarray:
    game_config = GameConfig(num_players=len(population),fps=FPS)
    agents = [NeuralNetworkAgent(config = game_config, weights=weights) for weights in population]

    total_scores = np.zeros(len(agents))
    N_RUNS = 3
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.starmap(
            run_single_game,
            [(agents, game_config) for _ in range(N_RUNS)]
        )
    for result in results:
        for idx, score in enumerate(result):
            total_scores[idx] += score
            
    average_scores = total_scores / 3
    print(f"Melhor: {np.max(average_scores):.2f} | Média: {np.mean(average_scores):.2f} | Std: {np.std(average_scores):.2f}")
    return average_scores

def train_and_test():
    print("\n--- Iniciando Treinamento com Algoritmo Genético ---")
    
    bats = BAT_Algorithm(
        population_size=POPULATION_SIZE,
        bats_lenght=27*32 + 24*32 + 24*3 + (32+ 24 +3),  # quantidade de pesos da rede
        fmin=0,
        fmax=10,
    )
    # print(bats.bats_lenght)
    # exit()
    best_weights_overall = None
    best_fitness_overall = -np.inf

    for generation in range(GENERATIONS):
        start_generation = time.time()
        current_best_weights, current_best_fitness  = bats.move_bats(game_fitness_function, generation)

        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_weights_overall = current_best_weights
            print(f'Backup generation -> Melhor Fitness Geral: {best_fitness_overall:.2f}')
            np.save("best_weights.npy", best_weights_overall)
        
        end = time.time()
        print(f"{generation + 1}/{GENERATIONS} Best Fitness: {current_best_fitness:.2f} Melhor Fitness Geral: {best_fitness_overall:.2f} ({end-start_generation:.2f} s)")
   

    print("\n--- Treinamento Concluído ---")
    print(f"Melhor Fitness Geral Alcançado: {best_fitness_overall:.2f}")

    if best_weights_overall is not None:
        np.save("best_weights.npy", best_weights_overall)
        print("Melhores pesos salvos em \'best_weights.npy\'")
 
        test_agent(best_weights_overall, num_tests=30, render=False)
    else:
        print("Nenhum peso ótimo encontrado.")

if __name__ == "__main__":
    train_and_test()