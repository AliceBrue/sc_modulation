""" functions to define weights optimisation algorithms """

from deap import base, creator, tools, algorithms, cma
import numpy as np
import pickle
import os
import random
import operator
import math


class Optim_Algo:

    def __init__(self, loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run=None):
        
        self.loss = loss
        self.n_weights = n_weights
        self.output_folder = output_folder
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.eps_bound = 2.e-5
        self.max_iter = max_iter
        self.pop_size = pop_size
        self.final_run = final_run

        # Create fitness type using DEAP
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        #creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))

        # Create the statistics object
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)

        # Create and initialise the toolbox
        self.toolbox = base.Toolbox()
        self.toolbox.register("evaluate", self.evaluate)
        
        # Initialize the algorithm
        self.checkpoint_file = self.output_folder + "checkpoint.pkl"
        if os.path.isfile(self.checkpoint_file):
            # Load the checkpoint data if any
            with open(self.checkpoint_file, "rb") as cp_file:
                cp = pickle.load(cp_file)
            self.population = cp["population"]
            self.start_gen = cp["generation"]
            self.halloffame = cp["halloffame"]
            self.logbook = cp["logbook"]
            random.setstate(cp["rndstate"])
            self.strategy = cp["strategy"]
        else:
            # Start a new evolution 
            self.population = None
            self.start_gen = 0
            self.halloffame = tools.HallOfFame(maxsize=1)
            self.logbook = tools.Logbook()
            self.logbook.header = ['gen', 'nevals'] + (self.stats.fields if self.stats else [])
            self.strategy = None

            # Initialize best_results.txt file
            with open(self.output_folder + "best_results.txt", "w") as best_results_file:
                best_results_file.write("Generation, Fitness, Params\n")
    

    def evaluate(self, individual):
        """Evaluation function for the individual."""
        return self.loss(individual)
    

    def checkBounds(self):
        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(len(child)):
                        if child[i] > self.max_weight:
                            child[i] = self.max_weight
                        elif child[i] < self.min_weight:
                            child[i] = self.min_weight
                return offspring
            return wrapper
        return decorator
    
    
    def closest_feasible(self, individual):
        """A function returning a valid individual from an invalid one."""
        feasible_ind = np.array(individual)
        feasible_ind = np.maximum(self.min_weight*np.ones(self.n_weights), feasible_ind)
        feasible_ind = np.minimum(self.max_weight*np.ones(self.n_weights), feasible_ind)
        return feasible_ind


    def recordings(self, generation):
        self.halloffame.update(self.population)
        record = self.stats.compile(self.population)
        self.logbook.record(gen=generation, nevals=len(self.population), **record)


    def store_data(self, generation):
        """Write the so far best fitness and parameters to the file"""
    
        best_weights = self.halloffame[0]
        best_fitness = best_weights.fitness.values[0]
        with open(self.output_folder + "best_results.txt", "a") as best_results_file:
            best_results_file.write(f"{generation}, {best_fitness}")
            for weight in best_weights:
                best_results_file.write(f", {weight}")
            best_results_file.write("\n")

        if generation % 10 == 0 or generation == self.max_iter - 1:
            # Write the data to the checkpoint file
            cp = dict(population=self.population, generation=generation, halloffame=self.halloffame,
                    logbook=self.logbook, rndstate=random.getstate(), strategy=self.strategy)

            with open(self.checkpoint_file, "wb") as cp_file:
                pickle.dump(cp, cp_file) 


    def record_and_store(self, generation):
        # Store the statistics of the current generation
        self.recordings(generation)

        # Store the checkpoint data and the best results
        self.store_data(generation)

        # Run the simulation with the best individual at the end of the optimisation
        if generation == self.max_iter - 1 and self.final_run is not None:
            self.final_run(self.halloffame[0])
    

    def run(self):
        raise NotImplementedError("You should implement this method in the subclass")
            


class CMA(Optim_Algo):
    
    def __init__(self, loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run):
        
        super().__init__(loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run)
        
        # Create fitness and individual types using DEAP
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        if os.path.isfile(self.checkpoint_file):
            # Load the checkpoint data if any
            with open(self.checkpoint_file, "rb") as cp_file:
                cp = pickle.load(cp_file)
            self.strategy = cp["strategy"]
        else:
            if not "baseline2" in output_folder and not "baseline3" in output_folder:
                np.random.seed(128)
            self.strategy = cma.Strategy(centroid=np.random.uniform(self.min_weight, self.max_weight, self.n_weights), sigma=1, lambda_=self.pop_size)        

        # Register the generator and update functions 
        self.toolbox.register("generate", self.strategy.generate, creator.Individual)
        self.toolbox.register("update", self.strategy.update) 
        
        # Add the decorator to check the bounds
        self.toolbox.decorate("generate", self.checkBounds())
        #self.toolbox.decorate("update", self.checkBounds())

    
    def run(self):
        '''if self.population is None:
            self.population = [creator.Individual(x) for x in (np.random.uniform(self.min_weight, self.max_weight, (self.strategy.mu, self.n_weights)))]
        for ind in self.population:
            ind.fitness.values = self.toolbox.evaluate(ind)'''

        for gen in range(self.start_gen, self.max_iter):
            self.population = self.toolbox.generate()                 

            # Evaluate the individuals in the population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, self.population)            ############ list(map(toolbox.evaluate, self.population))
            for ind, fit in zip(self.population, fitnesses):
                ind.fitness.values = fit

            # Update the strategy with the evaluated individuals
            self.toolbox.update(self.population)

            # Record and store the data
            self.record_and_store(gen)
                

class MuCommaLambda(Optim_Algo):
    
    def __init__(self, loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, cxpb, mutpb, final_run):
        
        super().__init__(loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run)
        self.lambda_ = pop_size
        self.mu = pop_size//2
        self.cxpb = cxpb
        self.mutpb = mutpb

        if self.lambda_ < self.mu:
            raise ValueError("lambda must be greater or equal to mu.")
        
        # Create fitness and individual types using DEAP
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        # Register the generator
        self.toolbox.register("weights_init", random.uniform, self.min_weight, self.max_weight)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.weights_init, self.n_weights)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register the genetic operators
        self.toolbox.register("select", tools.selBest, k=self.mu)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        
        # Add the decorator to check the bounds
        self.toolbox.decorate("mate", self.checkBounds())
        self.toolbox.decorate("mutate", self.checkBounds())


    def run(self):
        
        random.seed(42)
        if self.population is None:
            self.population = self.toolbox.population(n=self.mu)

        for gen in range(self.start_gen, self.max_iter):
            offspring = algorithms.varOr(self.population, self.toolbox, lambda_= self.lambda_, cxpb=self.cxpb, mutpb=self.mutpb)
            #offspring[:] = list(self.closest_feasible(offspring[:]))
            #offspring = map(self.closest_feasible, algorithms.varOr(self.population, self.toolbox, lambda_= self.lambda_, cxpb=self.cxpb, mutpb=self.mutpb))


            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)            ############ list(map(toolbox.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population from the offspring only
            self.population = self.toolbox.select(offspring)

            # Record and store the data
            self.record_and_store(gen)



class PSO(Optim_Algo):
    
    def __init__(self, loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run):
        
        super().__init__(loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run)

        # Create the individual type
        creator.create("Particle", list, fitness=creator.FitnessMin, speed=list, smin=None, smax=None, best=None)
        
        # Register the generator and update functions
        self.toolbox.register("particle", self.generate, smin=-(self.max_weight-self.min_weight)/8, smax=(self.max_weight-self.min_weight)/8)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.particle)
        self.toolbox.register("update", self.updateParticle)
        

    def generate(self, smin, smax):
        
        part = creator.Particle(random.uniform(self.min_weight, self.max_weight) for _ in range(self.n_weights)) 
        part.speed = [random.uniform(smin, smax) for _ in range(self.n_weights)]
        part.smin = smin
        part.smax = smax
        return part


    def updateParticle(self, part, best):
        
        u1 = (random.uniform(0, 1.0) for _ in range(len(part)))
        u2 = (random.uniform(0, 1.0) for _ in range(len(part)))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, best, part))
        part.speed = list(map(operator.add, part.speed, map(operator.add, v_u1, v_u2)))
        for i, speed in enumerate(part.speed):
            if abs(speed) < part.smin:
                part.speed[i] = math.copysign(part.smin, speed)
            elif abs(speed) > part.smax:
                part.speed[i] = math.copysign(part.smax, speed)
        #part[:] = list(map(operator.add, part, part.speed))
        part[:] = self.closest_feasible(list(map(operator.add, part, part.speed)))
        return part


    def run(self):
        
        if self.population is None:
            self.population = self.toolbox.population(n=self.pop_size)
        
        best = None
        for gen in range(self.start_gen, self.max_iter):
            for part in self.population:
                part.fitness.values = self.toolbox.evaluate(part)
                if not part.best or part.best.fitness < part.fitness:
                    part.best = creator.Particle(part)
                    part.best.fitness.values = part.fitness.values
                if not best or best.fitness < part.fitness:
                    best = creator.Particle(part)
                    best.fitness.values = part.fitness.values
            for part in self.population:
                part = self.toolbox.update(part, best)

            # Record and store the data
            self.record_and_store(gen)



class DE(Optim_Algo):  # Differential Evolution
    
    def __init__(self, loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run, CR = 0.25, F = 1):
        
        super().__init__(loss, n_weights, max_weight, min_weight, max_iter, pop_size, output_folder, final_run)
        self.F = F
        self.CR = CR
        
        # Create fitness and individual types using DEAP
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # Register the generator
        self.toolbox.register("weights_init", random.uniform, self.min_weight, self.max_weight)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.weights_init, self.n_weights)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register the genetic operators
        self.toolbox.register("select", tools.selBest, k=3)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        
        # Add the decorator to check the bounds
        self.toolbox.decorate("mate", self.checkBounds())
        self.toolbox.decorate("mutate", self.checkBounds())


    def run(self):
        
        if self.population is None:
            self.population = self.toolbox.population(n=self.pop_size)

        for gen in range(self.start_gen, self.max_iter):
            for i, ind in enumerate(self.population):
                a, b, c = self.toolbox.select(self.population)
                y = self.toolbox.clone(ind)
                index = random.randrange(self.n_weights)
                for j, value in enumerate(ind):
                    if j == index or random.random() < self.CR:
                        y[j] = self.closest_feasible(a[j] + self.F * (b[j] - c[j]))
                        # y[j] = a[j] + self.F * (b[j] - c[j])
                y.fitness.values = self.toolbox.evaluate(y)
                if y.fitness < ind.fitness:
                    self.population[i] = y

            # Record and store the data
            self.record_and_store(gen)

