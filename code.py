import random
import cv2
import numpy as np

# Constants
POPULATION_SIZE = 20
MUTATION_RATE = 0.01
CONTAINER_WIDTH = 800
CONTAINER_HEIGHT = 800

shapes = np.array([
    [70, 100],
    [60, 80],
    [100, 70],
    [80, 60],
    [120, 100],
    [100, 120],
    [150, 100],
    [100, 150],
    [200, 150],
    [150, 200]
])

colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255)]

"population of containers with random shapes"
def initialize_population():
  
    population = []
    for _ in range(POPULATION_SIZE):
        container = np.zeros((CONTAINER_WIDTH, CONTAINER_HEIGHT, 3), dtype=np.uint8)
        shapes_order = np.random.permutation(len(shapes))
        for i in shapes_order:
            shape = shapes[i]
            x = random.randint(0, CONTAINER_WIDTH - shape[0])
            y = random.randint(0, CONTAINER_HEIGHT - shape[1])
            container[y:y+shape[1], x:x+shape[0]] = colors[i % len(colors)]
            cv2.putText(container, str(i), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            population.append(container)
    return population

"calculating the total area of empty space in the container"
def fitness(container):
    
    gray = cv2.cvtColor(container, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = 0
    for contour in contours:
        area += cv2.contourArea(contour)
    return area

" Select two parents from the population "
def select_parents(population):
   
    parent1 = random.choice(population)
    parent2 = random.choice(population)
    parent1_fitness = fitness(parent1)
    parent2_fitness = fitness(parent2)
    if parent1_fitness < parent2_fitness:
        return parent1
    else:
        return parent2
    
"Create a child container by combining the parents' genetic material."
def crossover(parent1, parent2):

    child = np.zeros((CONTAINER_WIDTH, CONTAINER_HEIGHT, 3), dtype=np.uint8)
    for i, shape in enumerate(shapes):
        if i % 2 == 0:
            x1 = random.randint(0, CONTAINER_WIDTH - shape[0])
            y1 = random.randint(0, CONTAINER_HEIGHT - shape[1])
            x2 = random.randint(0, CONTAINER_WIDTH - shape[0])
            y2 = random.randint(0, CONTAINER_HEIGHT - shape[1])
            child[y1:y1+shape[1], x1:x1+shape[0]] = parent1[y1:y1+shape[1], x1:x1+shape[0]]
            if parent2.shape[0] > CONTAINER_HEIGHT or parent2.shape[1] > CONTAINER_WIDTH:
                parent2 = cv2.resize(parent2, (CONTAINER_WIDTH, CONTAINER_HEIGHT), interpolation=cv2.INTER_AREA)
            child[y2:y2+shape[1], x2:x2+shape[0]] = parent2[y2:y2+shape[1], x2:x2+shape[0]]
    return child

" Mutate the container by randomly changing the position of a shape"
def mutate(container):

    shape_index = random.randint(0, len(shapes)-1)
    shape = shapes[shape_index]
    x = random.randint(0, CONTAINER_WIDTH - shape[0])
    y = random.randint(0, CONTAINER_HEIGHT - shape[1])
    container[y:y+shape[1], x:x+shape[0]] = colors[shape_index % len(colors)]
    cv2.putText(container, str(shape_index), (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return container

def optimize():
  
    population = initialize_population()
    for generation in range(100):
        print(f'Generation {generation + 1}')
        next_generation = []
        for i in range(POPULATION_SIZE):
            parent1 = select_parents(population)
            parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            next_generation.append(child)
        population = next_generation
        best_container = min(population, key=fitness)
        best_fitness = fitness(best_container)
        print(f'Best fitness: {best_fitness}')
        cv2.imshow('Best container', best_container)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    
optimize()
