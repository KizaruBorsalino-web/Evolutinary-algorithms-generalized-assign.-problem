import random
import math
import sys
from typing import List, Tuple

# Optional: numpy for speed
try:
    import numpy as np
    HAS_NUMPY = True
except Exception:
    HAS_NUMPY = False

# -------------------------------
# Helper: compute bits per gene
# -------------------------------
def bits_per_gene(m: int) -> int:
    return max(1, math.ceil(math.log(m, 2)))

# -------------------------------
# Helper: binary ↔ integer
# -------------------------------
def bits_to_int(bits: List[int]) -> int:
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val

def int_to_bits(x: int, width: int) -> List[int]:
    b = []
    for i in range(width):
        b.append((x >> (width - 1 - i)) & 1)
    return b

# -------------------------------
# GAP data loader
# -------------------------------
def load_gap_instances(file_path: str) -> List[Tuple]:
    """
    Reads a file containing multiple GAP instances in the format:
      T
      m n
      <m lines of n costs>
      <m lines of n resources>
      <1 line of m capacities>
    Returns list of tuples: (cost_matrix, resource_matrix, capacities_vector)
    """
    instances = []
    with open(file_path, 'r') as f:
        data = f.read().strip().split()
    it = iter(data)
    T = int(next(it))
    for _ in range(T):
        m = int(next(it)); n = int(next(it))
        cost = [[float(next(it)) for _ in range(n)] for _ in range(m)]
        resource = [[float(next(it)) for _ in range(n)] for _ in range(m)]
        capacities = [float(next(it)) for _ in range(m)]
        instances.append((cost, resource, capacities))
    return instances

# -------------------------------
# Greedy repair (shared)
# -------------------------------
def greedy_repair(cost, resource, capacities, assignment):
    m = len(cost)
    n = len(cost[0])
    cap = list(capacities)
    load = [0.0] * m
    fixed = assignment[:]
    for j in range(n):
        i = fixed[j]
        load[i] += resource[i][j]
    for i in range(m):
        if load[i] > cap[i] + 1e-9:
            jobs = [j for j in range(n) if fixed[j] == i]
            jobs.sort(key=lambda j: cost[i][j], reverse=True)
            for j in jobs:
                if load[i] <= cap[i] + 1e-9:
                    break
                best_b = None
                best_cost = -float('inf')
                for b in range(m):
                    if b == i: continue
                    if load[b] + resource[b][j] <= cap[b] + 1e-9:
                        if cost[b][j] > best_cost:
                            best_cost = cost[b][j]
                            best_b = b
                if best_b is None:
                    continue
                fixed[j] = best_b
                load[i] -= resource[i][j]
                load[best_b] += resource[best_b][j]
    return fixed

# -------------------------------
# Fitness function (shared)
# -------------------------------
def compute_fitness(cost, resource, capacities, assignment):
    m = len(cost)
    n = len(cost[0])
    total = 0.0
    usage = [0.0] * m
    for j in range(n):
        i = assignment[j]
        total += cost[i][j]
        usage[i] += resource[i][j]
    penalty = 0.0
    for i in range(m):
        if usage[i] > capacities[i] + 1e-9:
            penalty += (usage[i] - capacities[i])
    return total - 1000.0 * penalty

# -------------------------------
# Calculate actual cost (without penalty)
# -------------------------------
def calculate_actual_cost(cost, best_assign):
    """Calculate the actual total cost of assignment"""
    total = 0.0
    for j, i in enumerate(best_assign):
        total += cost[i][j]
    return total

# ================================
# SOLVER 1: BCGA (Binary-Coded GA)
# ================================
class BCGA:
    def __init__(self, cost, resource, capacities,
                 population_size=60,
                 generations=80,
                 gene_bits=None,
                 crossover_rate=0.8,
                 mutation_rate=0.05,
                 seed=42):
        self.cost = cost
        self.resource = resource
        self.capacities = capacities
        self.m, self.n = len(cost), len(cost[0])
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.seed = seed
        random.seed(seed)
        self.bits = gene_bits if gene_bits is not None else bits_per_gene(self.m)
        self.chrom_length = self.bits * self.n

    def random_chromosome(self):
        return [random.randint(0, 1) for _ in range(self.chrom_length)]

    def decode(self, chrom):
        assign = []
        for j in range(self.n):
            start = j * self.bits
            end = start + self.bits
            val = bits_to_int(chrom[start:end])
            assign.append(val % self.m)
        return assign

    def repair(self, assignment):
        return greedy_repair(self.cost, self.resource, self.capacities, assignment)

    def fitness(self, assignment):
        return compute_fitness(self.cost, self.resource, self.capacities, assignment)

    def select_tournament(self, pop, fits, k=3):
        idxs = random.sample(range(len(pop)), k)
        best = max(idxs, key=lambda idx: fits[idx])
        return pop[best]

    def crossover(self, a, b):
        if random.random() > self.crossover_rate:
            return a[:], b[:]
        point = random.randint(1, self.chrom_length-1)
        return a[:point] + b[point:], b[:point] + a[point:]

    def mutate(self, chrom):
        for i in range(self.chrom_length):
            if random.random() < self.mutation_rate:
                chrom[i] = 1 - chrom[i]
        return chrom

    def run(self, verbose=False):
        pop = [self.random_chromosome() for _ in range(self.population_size)]
        best_assign = None
        best_cost = -float('inf')
        history = []
        for gen in range(self.generations):
            fits = []
            for chrom in pop:
                dec = self.decode(chrom)
                rep = self.repair(dec)
                f = self.fitness(rep)
                fits.append(f)
            best_idx = max(range(len(fits)), key=lambda i: fits[i])
            best_val = fits[best_idx]
            history.append(best_val)
            if verbose and gen % 10 == 0:
                print(f"BCGA Gen {gen}: best fitness ~ {best_val:.4f}")
            new_pop = [pop[best_idx][:]]
            while len(new_pop) < self.population_size:
                p1 = self.select_tournament(pop, fits)
                p2 = self.select_tournament(pop, fits)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.extend([c1, c2])
            pop = new_pop[:self.population_size]
        # Final best
        fits = [self.fitness(self.repair(self.decode(ch))) for ch in pop]
        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        best_assign = self.repair(self.decode(pop[best_idx]))
        best_cost = fits[best_idx]
        return best_assign, best_cost, history

# ================================
# SOLVER 2: Real-Coded GA
# ================================
class RealGA:
    def __init__(self, cost, resource, capacities,
                 population_size=60,
                 generations=80,
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 mutation_strength=0.3,
                 seed=42):
        self.cost = cost
        self.resource = resource
        self.capacities = capacities
        self.m = len(cost)
        self.n = len(cost[0])
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.seed = seed
        random.seed(seed)
        self.domain_min = 0.0
        self.domain_max = float(self.m)

    def random_individual(self):
        return [random.uniform(self.domain_min, self.domain_max) for _ in range(self.n)]

    def map_to_assignment(self, x):
        return [int(xj) % self.m for xj in x]

    def repair(self, assignment):
        return greedy_repair(self.cost, self.resource, self.capacities, assignment)

    def fitness(self, assignment):
        return compute_fitness(self.cost, self.resource, self.capacities, assignment)

    def select_tournament(self, pop, fits, k=3):
        idxs = random.sample(range(len(pop)), k)
        best = max(idxs, key=lambda idx: fits[idx])
        return pop[best]

    def crossover(self, a, b):
        if random.random() > self.crossover_rate:
            return a[:], b[:]
        # Blend crossover
        c1, c2 = [], []
        for i in range(self.n):
            alpha = random.random()
            c1.append(alpha * a[i] + (1 - alpha) * b[i])
            c2.append(alpha * b[i] + (1 - alpha) * a[i])
        return c1, c2

    def mutate(self, x):
        for i in range(self.n):
            if random.random() < self.mutation_rate:
                delta = random.uniform(-self.mutation_strength, self.mutation_strength) * self.m
                x[i] += delta
                x[i] = max(self.domain_min, min(self.domain_max, x[i]))
        return x

    def run(self, verbose=False):
        pop = [self.random_individual() for _ in range(self.population_size)]
        best_assign = None
        best_cost = -float('inf')
        history = []
        for gen in range(self.generations):
            fits = []
            for x in pop:
                cand = self.map_to_assignment(x)
                rep = self.repair(cand)
                f = self.fitness(rep)
                fits.append(f)
            best_idx = max(range(len(fits)), key=lambda i: fits[i])
            best_val = fits[best_idx]
            history.append(best_val)
            if verbose and gen % 10 == 0:
                print(f"RealGA Gen {gen}: best fitness ~ {best_val:.4f}")
            new_pop = [pop[best_idx][:]]
            while len(new_pop) < self.population_size:
                p1 = self.select_tournament(pop, fits)
                p2 = self.select_tournament(pop, fits)
                c1, c2 = self.crossover(p1, p2)
                c1 = self.mutate(c1)
                c2 = self.mutate(c2)
                new_pop.extend([c1, c2])
            pop = new_pop[:self.population_size]
        # Final best
        fits = [self.fitness(self.repair(self.map_to_assignment(x))) for x in pop]
        best_idx = max(range(len(fits)), key=lambda i: fits[i])
        best_assign = self.repair(self.map_to_assignment(pop[best_idx]))
        best_cost = fits[best_idx]
        return best_assign, best_cost, history

# ================================
# SOLVER 3: PSO (Particle Swarm Optimization)
# ================================
class PSO:
    def __init__(self, cost, resource, capacities,
                 population_size=60,
                 iterations=80,
                 w=0.7,
                 c1=1.5,
                 c2=1.5,
                 seed=42):
        self.cost = cost
        self.resource = resource
        self.capacities = capacities
        self.m = len(cost)
        self.n = len(cost[0])
        self.population_size = population_size
        self.iterations = iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        random.seed(seed)
        self.domain_min = 0.0
        self.domain_max = float(self.m)

    def random_particle(self):
        return [random.uniform(self.domain_min, self.domain_max) for _ in range(self.n)]

    def random_velocity(self):
        return [random.uniform(-1.0, 1.0) for _ in range(self.n)]

    def map_to_assignment(self, x):
        return [int(xj) % self.m for xj in x]

    def repair(self, assignment):
        return greedy_repair(self.cost, self.resource, self.capacities, assignment)

    def fitness(self, assignment):
        return compute_fitness(self.cost, self.resource, self.capacities, assignment)

    def run(self, verbose=False):
        # Initialize particles
        particles = [self.random_particle() for _ in range(self.population_size)]
        velocities = [self.random_velocity() for _ in range(self.population_size)]
        pbest_positions = [p[:] for p in particles]
        pbest_scores = [-float('inf')] * self.population_size
        
        gbest_position = None
        gbest_score = -float('inf')
        history = []

        for iteration in range(self.iterations):
            for i in range(self.population_size):
                cand = self.map_to_assignment(particles[i])
                rep = self.repair(cand)
                score = self.fitness(rep)
                
                if score > pbest_scores[i]:
                    pbest_scores[i] = score
                    pbest_positions[i] = particles[i][:]
                
                if score > gbest_score:
                    gbest_score = score
                    gbest_position = particles[i][:]
            
            history.append(gbest_score)
            if verbose and iteration % 10 == 0:
                print(f"PSO Iter {iteration}: best fitness ~ {gbest_score:.4f}")
            
            # Update velocities and positions
            for i in range(self.population_size):
                for j in range(self.n):
                    r1 = random.random()
                    r2 = random.random()
                    velocities[i][j] = (self.w * velocities[i][j] +
                                        self.c1 * r1 * (pbest_positions[i][j] - particles[i][j]) +
                                        self.c2 * r2 * (gbest_position[j] - particles[i][j]))
                    particles[i][j] += velocities[i][j]
                    particles[i][j] = max(self.domain_min, min(self.domain_max, particles[i][j]))
        
        best_assign = self.repair(self.map_to_assignment(gbest_position))
        return best_assign, gbest_score, history

# ================================
# SOLVER 4: TLBO (Teaching-Learning Based Optimization)
# ================================
class TLBO:
    def __init__(self, cost, resource, capacities,
                 population_size=60,
                 generations=80,
                 seed=42):
        self.cost = cost
        self.resource = resource
        self.capacities = capacities
        self.m = len(cost)
        self.n = len(cost[0])
        self.population_size = population_size
        self.generations = generations
        self.seed = seed
        random.seed(seed)
        self.domain_min = 0.0
        self.domain_max = float(self.m)

    def random_population(self):
        return [
            [random.uniform(self.domain_min, self.domain_max) for _ in range(self.n)]
            for __ in range(self.population_size)
        ]

    def map_to_assignment(self, x):
        return [int(xj) % self.m for xj in x]

    def repair(self, assignment):
        return greedy_repair(self.cost, self.resource, self.capacities, assignment)

    def fitness(self, assignment):
        return compute_fitness(self.cost, self.resource, self.capacities, assignment)

    def teacher_phase(self, population, fitnesses):
        best_idx = max(range(len(population)), key=lambda k: fitnesses[k])
        teacher = population[best_idx]
        n = self.n
        mean_vec = [0.0]*n
        for x in population:
            for j in range(n):
                mean_vec[j] += x[j]
        mean_vec = [v / len(population) for v in mean_vec]
        for idx in range(len(population)):
            x = population[idx]
            for j in range(n):
                r = random.random()
                x[j] = x[j] + r * (teacher[j] - mean_vec[j])
                x[j] = max(self.domain_min, min(self.domain_max, x[j]))

    def learner_phase(self, population, fitnesses):
        size = len(population)
        for _ in range(size // 2):
            a_idx, b_idx = random.sample(range(size), 2)
            a = population[a_idx]
            b = population[b_idx]
            if fitnesses[a_idx] >= fitnesses[b_idx]:
                for j in range(self.n):
                    r = random.random()
                    b[j] = b[j] + r * (a[j] - b[j])
                    b[j] = max(self.domain_min, min(self.domain_max, b[j]))
                population[b_idx] = b
            else:
                for j in range(self.n):
                    r = random.random()
                    a[j] = a[j] + r * (b[j] - a[j])
                    a[j] = max(self.domain_min, min(self.domain_max, a[j]))
                population[a_idx] = a

    def run(self, verbose=False):
        pop = self.random_population()
        best_assign = None
        best_cost = -float('inf')
        history = []

        for gen in range(self.generations):
            fitnesses = []
            for x in pop:
                cand = self.map_to_assignment(x)
                rep = self.repair(cand)
                f = self.fitness(rep)
                fitnesses.append(f)

            self.teacher_phase(pop, fitnesses)

            fitnesses = []
            for x in pop:
                cand = self.map_to_assignment(x)
                rep = self.repair(cand)
                f = self.fitness(rep)
                fitnesses.append(f)

            self.learner_phase(pop, fitnesses)

            best_in_gen = -float('inf')
            best_in_gen_assign = None
            for x in pop:
                cand = self.map_to_assignment(x)
                rep = self.repair(cand)
                f = self.fitness(rep)
                if f > best_in_gen:
                    best_in_gen = f
                    best_in_gen_assign = rep
            if best_in_gen > best_cost:
                best_cost = best_in_gen
                best_assign = best_in_gen_assign

            history.append(best_cost)
            if verbose and gen % 10 == 0:
                print(f"TLBO Gen {gen}: best fitness ~ {best_cost:.4f}")

        return best_assign, best_cost, history

# ================================
# SOLVER 5: DE (Differential Evolution)
# ================================
class DE:
    def __init__(self, cost, resource, capacities,
                 population_size=60,
                 generations=80,
                 F=0.8,
                 CR=0.9,
                 seed=42):
        self.cost = cost
        self.resource = resource
        self.capacities = capacities
        self.m = len(cost)
        self.n = len(cost[0])
        self.population_size = population_size
        self.generations = generations
        self.F = F
        self.CR = CR
        self.seed = seed
        random.seed(seed)
        self.domain_min = 0.0
        self.domain_max = float(self.m)

    def random_individual(self):
        return [random.uniform(self.domain_min, self.domain_max) for _ in range(self.n)]

    def map_to_assignment(self, x):
        return [int(xj) % self.m for xj in x]

    def repair(self, assignment):
        return greedy_repair(self.cost, self.resource, self.capacities, assignment)

    def fitness(self, assignment):
        return compute_fitness(self.cost, self.resource, self.capacities, assignment)

    def run(self, verbose=False):
        pop = [self.random_individual() for _ in range(self.population_size)]
        best_assign = None
        best_cost = -float('inf')
        history = []

        for gen in range(self.generations):
            new_pop = []
            for i in range(self.population_size):
                # Mutation
                indices = [j for j in range(self.population_size) if j != i]
                a, b, c = random.sample(indices, 3)
                mutant = [pop[a][j] + self.F * (pop[b][j] - pop[c][j]) for j in range(self.n)]
                mutant = [max(self.domain_min, min(self.domain_max, v)) for v in mutant]
                
                # Crossover
                trial = []
                for j in range(self.n):
                    if random.random() < self.CR:
                        trial.append(mutant[j])
                    else:
                        trial.append(pop[i][j])
                
                # Selection
                cand_trial = self.map_to_assignment(trial)
                rep_trial = self.repair(cand_trial)
                f_trial = self.fitness(rep_trial)
                
                cand_curr = self.map_to_assignment(pop[i])
                rep_curr = self.repair(cand_curr)
                f_curr = self.fitness(rep_curr)
                
                if f_trial > f_curr:
                    new_pop.append(trial)
                    if f_trial > best_cost:
                        best_cost = f_trial
                        best_assign = rep_trial
                else:
                    new_pop.append(pop[i])
                    if f_curr > best_cost:
                        best_cost = f_curr
                        best_assign = rep_curr
            
            pop = new_pop
            history.append(best_cost)
            if verbose and gen % 10 == 0:
                print(f"DE Gen {gen}: best fitness ~ {best_cost:.4f}")

        return best_assign, best_cost, history

# -------------------------------
# Output helper
# -------------------------------
def format_assignment_output(cost, best_assign):
    """
    Returns a string with the three-line output format:
      - Assignment Matrix (m rows, n columns, 0/1)
      - Assigned Costs (space-separated)
      - Total Cost
    """
    m = len(cost)
    n = len(cost[0])
    
    matrix = [[0 for _ in range(n)] for __ in range(m)]
    for j, i in enumerate(best_assign):
        matrix[i][j] = 1
    
    lines = []
    for i in range(m):
        lines.append(" ".join(str(matrix[i][j]) for j in range(n)))
    
    assigned_costs = [cost[best_assign[j]][j] for j in range(n)]
    costs_line = " ".join(f"{c:.4f}" for c in assigned_costs)
    
    total_cost = sum(cost[best_assign[j]][j] for j in range(n))
    total_line = f"{total_cost:.4f}"
    
    return "\n".join(lines) + "\n" + costs_line + "\n" + total_line

# -------------------------------
# Main driver - runs all solvers with statistics
# -------------------------------
def main():
    """Run all solvers and save to separate output files with statistics"""
    solvers_config = {
        'bcga': BCGA,
        'realga': RealGA,
        'pso': PSO,
        'tlbo': TLBO,
        'de': DE
    }
    
    file_paths = ["gap_sample_data_txt.txt",
        "gap1.txt", "gap2.txt", "gap3.txt", "gap4.txt", 
        "gap5.txt", "gap6.txt", "gap7.txt", "gap8.txt",
        "gap9.txt", "gap10.txt", "gap11.txt", "gap12.txt"
    ]
    
    # Dictionary to store average costs: solver -> {file -> avg_cost}
    all_statistics = {}
    
    print("=" * 70)
    print("GAP Solver - Running All Algorithms with Statistics")
    print("=" * 70)
    
    for solver_name, SolverClass in solvers_config.items():
        print(f"\n[{solver_name.upper()}] Starting...")
        output_file = f"{solver_name}_output.txt"
        
        file_statistics = {}  # Store stats for this solver
        
        with open(output_file, "w") as out:
            for f in file_paths:
                try:
                    instances = load_gap_instances(f)
                except FileNotFoundError:
                    print(f"  Warning: {f} not found, skipping...")
                    continue
                
                file_costs = []  # Collect costs for all instances in this file
                
                for idx, (cost, resource, capacities) in enumerate(instances, start=1):
                    seed = 123 + idx
                    
                    # Create solver instance
                    if solver_name == 'pso':
                        solver = SolverClass(cost, resource, capacities,
                                           population_size=60, iterations=80, seed=seed)
                    else:
                        solver = SolverClass(cost, resource, capacities,
                                           population_size=60, generations=80, seed=seed)
                    
                    # Run solver
                    best_assign, best_cost, history = solver.run(verbose=False)
                    
                    # Calculate actual cost (without penalty)
                    actual_cost = calculate_actual_cost(cost, best_assign)
                    file_costs.append(actual_cost)
                    
                    # Write output
                    output_block = format_assignment_output(cost, best_assign)
                    out.write(output_block)
                    out.write("\n\n")
                
                # Calculate average for this file
                if file_costs:
                    avg_cost = sum(file_costs) / len(file_costs)
                    file_statistics[f] = avg_cost
        
        all_statistics[solver_name] = file_statistics
        print(f"[{solver_name.upper()}] ✓ Results written to {output_file}")
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("AVERAGE COST SUMMARY (Mean across all instances per file)")
    print("=" * 70)
    
    # Create summary file
    with open("average_costs_summary.txt", "w") as summary_file:
        summary_file.write("=" * 70 + "\n")
        summary_file.write("AVERAGE COST SUMMARY (Mean across all instances per file)\n")
        summary_file.write("=" * 70 + "\n\n")
        
        for solver_name in solvers_config.keys():
            print(f"\n{solver_name.upper()}:")
            summary_file.write(f"\n{solver_name.upper()}:\n")
            summary_file.write("-" * 70 + "\n")
            
            for f in file_paths:
                if f in all_statistics[solver_name]:
                    avg_cost = all_statistics[solver_name][f]
                    line = f"  {f:12s} -> Average Cost: {avg_cost:10.4f}"
                    print(line)
                    summary_file.write(line + "\n")
            
            # Calculate overall average for this solver
            solver_costs = list(all_statistics[solver_name].values())
            if solver_costs:
                overall_avg = sum(solver_costs) / len(solver_costs)
                overall_line = f"  {'OVERALL':12s} -> Average Cost: {overall_avg:10.4f}"
                print(f"\n{overall_line}")
                summary_file.write(f"\n{overall_line}\n")
    
    print("\n" + "=" * 70)
    print("All algorithms completed successfully!")
    print("=" * 70)
    print("\nOutput files created:")
    for solver_name in solvers_config.keys():
        print(f"  • {solver_name}_output.txt")
    print(f"  • average_costs_summary.txt (Summary of all averages)")
    print()

if __name__ == "__main__":
    main()
