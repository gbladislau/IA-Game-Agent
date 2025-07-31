import numpy as np

class BAT_Algorithm:
    def __init__(self, population_size: int, bats_lenght: int, fmin:int, fmax:int, A_min=1, A_max=2, gamma=0.9, alpha=0.9) -> None:
        self.population_size = population_size
        self.bats_lenght = bats_lenght          #  weights of the network
        
        self.positions = np.random.rand(population_size, bats_lenght) * 2 - 1  # valores entre -1 e 1 --> X
        self.velocity = np.random.rand(population_size, bats_lenght) * 2 - 1   # valores entre -1 e 1 --> V
        self.frequencies = np.random.uniform(low=fmin, high=fmax, size=population_size)
        self.loudness = np.random.uniform(low=A_min, high=A_max, size=population_size)
        self.rate = np.random.uniform(low=0.0001, high=0.1, size=population_size)
        self.initial_rate = self.rate
        self.gamma = gamma
        self.alpha = alpha
    
    def is_conv(self): # ficou preso em um minimo local
       return (self.positions == self.positions[0]).all()
    
    def move_bats(self, game_fitness_function, iteration):
        fitness_scores = game_fitness_function(self.positions)
        
        best_fitness = fitness_scores.max()
        best_individual_idx = np.argmax(fitness_scores)
        best_position = self.positions[best_individual_idx]
        
        if self.is_conv():
            print("----- CONVERGIU -----")
        # Update pulse freq
        betha = np.random.uniform(low=0,high=1,size=self.population_size)
        self.frequencies = min(self.frequencies) + (max(self.frequencies) - min(self.frequencies)) * betha
        # update velocity
        self.velocity = self.velocity + (self.positions - best_position) * self.frequencies[:, np.newaxis]
        # update pos 
        temporary_positions = self.positions + self.velocity
        
        for i in range(0, self.population_size): # generate new local solution arround the best solution
            if np.random.rand() > self.rate[i]:
               temporary_positions[i] = best_position + np.random.uniform(low=-1, high=1, size=best_position.shape[0]) * self.loudness.mean()
            if np.random.rand() < self.loudness[i] and fitness_scores[i] < best_fitness: # accept new solutions with a certain probability
                self.positions[i] = np.clip(temporary_positions[i], -1, 1)
                self.rate[i] = self.initial_rate[i]*(1-np.exp(-self.gamma*iteration))
                self.loudness[i] = self.alpha*self.loudness[i]
             
        return self.positions[best_individual_idx], fitness_scores[best_individual_idx]