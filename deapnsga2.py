import array
import random
import numpy
import matplotlib.pyplot as plt
from math import sqrt
from deap import algorithms, base, benchmarks, creator, tools

creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)
toolbox = base.Toolbox()

low = 0.0
high = 1.0
dimensions = 30

def uniform(lower, upper, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(lower, upper)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([lower] * size, [upper] * size)]

toolbox.register("attr_float", uniform, low, high, dimensions)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", benchmarks.zdt1)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=low, up=high, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=low, up=high, eta=20.0, indpb=1.0/dimensions)
toolbox.register("select", tools.selNSGA2)

def EMO(gen):
    INDIV = 100
    CXPB = 0.9
    NGEN = gen

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    pop = toolbox.population(n=INDIV)

    # applying fitness to people with invalid fitnesses
    invalid_individual = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_individual)
    for ind, fit in zip(invalid_individual, fitnesses):
        ind.fitness.values = fit

    # assign crowding distance
    pop = toolbox.select(pop, len(pop))

    # go through generations
    for gen in range(1, NGEN):

        # selection using dominance
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # applying crossover
        for indiv1, indiv2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= CXPB:
                toolbox.mate(indiv1, indiv2)

            #mutates based on indp prob (1/dimensions)
            toolbox.mutate(indiv1)
            toolbox.mutate(indiv2)
            del indiv1.fitness.values, indiv2.fitness.values

        # evaluating fitness of individuals with invalid fitnesses
        invalid_individual = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_individual)
        for ind, fit in zip(invalid_individual, fitnesses):
            ind.fitness.values = fit

        # Chossing a population for the next generation
        pop = toolbox.select(pop + offspring, INDIV)


    return pop

if __name__ == '__main__':

    #pop around 20 shows best multiple fronts for zdt1
    #pops 10, 20, 30, 40, 50 show great transition from convex to concave for zdt2
    # zdt3 is 0-5; gets different fronts quickly
    # zdt 6 - 60 to 100

    print "ZDT1"
    pop1 = EMO(250)
    #print arr
    pop1.sort(key=lambda x: x.fitness.values)

    front1 = numpy.array([ind.fitness.values for ind in pop1])
    plt.scatter(front1[:,0], front1[:,1], c="b")
    plt.axis("tight")
    plt.show()

    # print "ZDT2 GEN:30"
    # pop2 = EMO(30)
    # pop2.sort(key=lambda x: x.fitness.values)

    # front2 = numpy.array([ind.fitness.values for ind in pop2])
    # plt.scatter(front2[:,0], front2[:,1], c="b")
    # plt.axis("tight")
    # plt.show()

    # print "ZDT2 GEN:45"
    # pop3 = EMO(45)
    # pop3.sort(key=lambda x: x.fitness.values)

    # front3 = numpy.array([ind.fitness.values for ind in pop3])
    # plt.scatter(front3[:,0], front3[:,1], c="b")
    # plt.axis("tight")
    # plt.show()

    # print "ZDT2 GEN:60"
    # pop4 = EMO(60)
    # pop4.sort(key=lambda x: x.fitness.values)

    # front4 = numpy.array([ind.fitness.values for ind in pop4])
    # plt.scatter(front4[:,0], front4[:,1], c="b")
    # plt.axis("tight")
    # plt.show()
