import time
import logging
import itertools
import math
import copy
import pdb
from random import Random 
import numpy as np

import matplotlib.pyplot as plt

def reduce_archive(archive, N=10):

    # store objectives, then DVs, then index
    num_DV = len(archive[0].candidate)
    num_Obj = len(archive[0].fitness)
    pf = np.zeros([len(archive),num_DV+num_Obj+1])
    for ii in range(len(archive)):
        pf[ii,0:num_Obj] = archive[ii].fitness
        pf[ii,num_Obj:num_Obj+num_DV] = archive[ii].candidate
        pf[ii,-1] = ii

    inds = np.argsort(pf[:,1],axis=0)
    pf = pf[inds[:],:]

    # first integrate total distance over first two objectives
    tot_length = 0
    lengths = np.zeros(pf.shape[0])
    for ii in range(1,pf.shape[0]):
        dx = pf[ii,0] - pf[ii-1,0]
        dy = pf[ii,1] - pf[ii-1,1]
        ds = math.sqrt(dx**2 + dy**2)
        tot_length += ds
        lengths[ii] = tot_length # distance of each point along the pareto front

    # find as close to N evenly spaced points as possible
    pf_red = np.zeros([N,pf.shape[1]])
    jj = 0
    Ni = lengths.size
    for ii in range(Ni):
        if jj == N: break
        if (Ni - ii) == (N - jj):
            pf_red[jj:N+1,:] = pf[ii:Ni+1,:]
            break
        elif lengths[ii] >= jj*tot_length/float(N-1):
            pf_red[jj,:] = pf[ii,:]
            jj += 1

    #plt.figure()
    #plt.scatter(pf[:,0],pf[:,1],color='blue',s=20)
    #plt.scatter(pf_red[:,0],pf_red[:,1],color='red',s=10)
    #plt.show()

    archive_reduced = []
    for jj in range(pf_red.shape[0]):
        archive_reduced.append(archive[int(pf_red[jj,-1])])

    return archive_reduced

class Individual(object):
    """Represents an individual in an evolutionary computation.
    
    An individual is defined by its candidate solution and the
    fitness (or value) of that candidate solution.
    
    Public Attributes:
    
    - *candidate* -- the candidate solution
    - *fitness* -- the value of the candidate solution
    - *birthdate* -- the system time at which the individual was created
    - *maximize* -- Boolean value stating use of maximization
    
    """
    def __init__(self, candidate=None, maximize=True):
        self.candidate = candidate
        self.fitness = None
        self.birthdate = time.time()
        self.maximize = maximize
        self.sigma = 0.1
        self.alpha = 0.0
    
    def __setattr__(self, name, val):
        if name == 'candidate':
            self.__dict__[name] = val
            self.fitness = None
        else:
            self.__dict__[name] = val
    
    def __str__(self):
        return '%s : %s' % (str(self.candidate), str(self.fitness))
        
    def __repr__(self):
        return '<Individual: candidate = %s, fitness = %s, birthdate = %s>' % ( str(self.candidate), str(self.fitness), self.birthdate )
        
    def __lt__(self, other):
        if self.fitness is not None and other.fitness is not None:
            if self.maximize: 
                return self.fitness < other.fitness
            else:
                return self.fitness > other.fitness
        else:
            raise Exception('fitness is not defined')

    def __le__(self, other):
        return self < other or not other < self
            
    def __gt__(self, other):
        if self.fitness is not None and other.fitness is not None:
            return other < self
        else:
            raise Exception('fitness is not defined')

    def __ge__(self, other):
        return other < self or not self < other
        
    def __lshift__(self, other):
        return self < other
    
    def __rshift__(self, other):
        return other < self
        
    def __ilshift__(self, other):
        raise TypeError("unsupported operand type(s) for <<=: 'Individual' and 'Individual'")
    
    def __irshift__(self, other):
        raise TypeError("unsupported operand type(s) for >>=: 'Individual' and 'Individual'")
        
    def __eq__(self, other):
        return self.candidate == other.candidate
        
    def __ne__(self, other):
        return self.candidate != other.candidate


class Pareto(object):
    """Represents a Pareto multiobjective solution.
    
    A Pareto solution is a multiobjective value that can be compared
    to other Pareto values using Pareto preference. This means that
    a solution dominates, or is better than, another solution if it is
    better than or equal to the other solution in all objectives and
    strictly better in at least one objective.
    
    Since some problems may mix maximization and minimization among
    different objectives, an optional `maximize` parameter may be
    passed upon construction of the Pareto object. This parameter
    may be a list of Booleans of the same length as the set of 
    objective values. If this parameter is used, then the `maximize`
    parameter of the evolutionary computation's `evolve` method 
    should be left as the default True value in order to avoid
    confusion. (Setting the `evolve`'s parameter to False would
    essentially invert all of the Booleans in the Pareto `maximize`
    list.) So, if all objectives are of the same type (either
    maximization or minimization), then it is best simply to use
    the `maximize` parameter of the `evolve` method and to leave
    the `maximize` parameter of the Pareto initialization set to
    its default True value. However, if the objectives are mixed
    maximization and minimization, it is best to leave the `evolve`'s
    `maximize` parameter set to its default True value and specify
    the Pareto's `maximize` list to the appropriate Booleans.
    
    """
    def __init__(self, values=[], maximize=True):
        self.values = values
        try:
            iter(maximize)
        except TypeError:
            maximize = [maximize for v in values]
        self.maximize = maximize
        
    def __len__(self):
        return len(self.values)
    
    def __getitem__(self, key):
        return self.values[key]
        
    def __iter__(self):
        return iter(self.values)
    
    def __lt__(self, other):
        if len(self.values) != len(other.values):
            raise NotImplementedError
        else:
            not_worse = True
            strictly_better = False
            for x, y, m in zip(self.values, other.values, self.maximize):
                if m:
                    if x > y:
                        not_worse = False
                    elif y > x:
                        strictly_better = True
                else:
                    if x < y:
                        not_worse = False
                    elif y < x:
                        strictly_better = True
            return not_worse and strictly_better
            
    def __le__(self, other):
        return self < other or not other < self
        
    def __gt__(self, other):
        return other < self
        
    def __ge__(self, other):
        return other < self or not self < other
        
    def __eq__(self, other):
        return self.values == other.values
    
    def __str__(self):
        return str(self.values)
        
    def __repr__(self):
        return str(self.values)

class Bounder(object):
    """Defines a basic bounding function for numeric lists.
    
    This callable class acts as a function that bounds a 
    numeric list between the lower and upper bounds specified.
    These bounds can be single values or lists of values. For
    instance, if the candidate is composed of five values, each
    of which should be bounded between 0 and 1, you can say
    ``Bounder([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])`` or just
    ``Bounder(0, 1)``. If either the ``lower_bound`` or 
    ``upper_bound`` argument is ``None``, the Bounder leaves 
    the candidate unchanged (which is the default behavior).
    
    A bounding function is necessary to ensure that all 
    evolutionary operators respect the legal bounds for 
    candidates. If the user is using only custom operators
    (which would be aware of the problem constraints), then 
    those can obviously be tailored to enforce the bounds
    on the candidates themselves. But the built-in operators
    make only minimal assumptions about the candidate solutions.
    Therefore, they must rely on an external bounding function
    that can be user-specified (so as to contain problem-specific
    information). As a historical note, ECsPy was originally 
    designed to require the maximum and minimum values for all
    components of the candidate solution to be passed to the
    ``evolve`` method. However, this was replaced by the bounding
    function approach because it made fewer assumptions about
    the structure of a candidate (e.g., that candidates were 
    going to be lists) and because it allowed the user the
    flexibility to provide more elaborate boundings if needed.
    
    In general, a user-specified bounding function must accept
    two arguments: the candidate to be bounded and the keyword
    argument dictionary. Typically, the signature of such a 
    function would be ``bounding_function(candidate, args)``.
    This function should return the resulting candidate after 
    bounding has been performed.
    
    Public Attributes:
    
    - *lower_bound* -- the lower bound for a candidate
    - *upper_bound* -- the upper bound for a candidate
    
    """
    def __init__(self, lower_bound=None, upper_bound=None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        if self.lower_bound is not None and self.upper_bound is not None:
            try:
                iter(self.lower_bound)
            except TypeError:
                self.lower_bound = itertools.repeat(self.lower_bound)
            try:
                iter(self.upper_bound)
            except TypeError:
                self.upper_bound = itertools.repeat(self.upper_bound)
            

    def __call__(self, candidate, args):
        # The default would be to leave the candidate alone
        # unless both bounds are specified.
        if self.lower_bound is None or self.upper_bound is None:
            return candidate
        else:
            try:
                iter(self.lower_bound)
            except TypeError:
                self.lower_bound = [self.lower_bound] * len(candidate)
            try:
                iter(self.upper_bound)
            except TypeError:
                self.upper_bound = [self.upper_bound] * len(candidate)
            bounded_candidate = copy.copy(candidate)
            for i, (c, lo, hi) in enumerate(zip(candidate, self.lower_bound, self.upper_bound)):
                bounded_candidate[i] = max(min(c, hi), lo)
            return bounded_candidate



def generation_termination(population, num_generations, num_evaluations, args):
    """Return True if the number of generations meets or exceeds a maximum.
    
    This function compares the number of generations with a specified 
    maximum. It returns True if the maximum is met or exceeded.

    Optional keyword arguments in args:
    
    *max_generations* -- the maximum generations (default 1) 
    
    """
    max_generations = args.setdefault('max_generations', 1)
    if num_generations >= max_generations:
        return True
    return False
 

def best_archiver_mfuq(random, population, archive, args):
    """Archive only the best individual(s).
    
    This function archives the best solutions and removes inferior ones.
    If the comparison operators have been overloaded to define Pareto
    preference (as in the ``Pareto`` class), then this archiver will form 
    a Pareto archive.
    
    """
    new_archive = archive
    for ind in population:
        if len(new_archive) == 0:
            new_archive.append(ind)
        else:
            should_remove = []
            should_add = True
            for a in new_archive:
                if ind == a:
                    should_add = False
                    break
                elif ind < a:
                    should_add = False
                elif ind > a:
                    should_remove.append(a)
            for r in should_remove:
                new_archive.remove(r)
            if should_add:
                new_archive.append(ind)
    return new_archive

# lwc24 method
def PF_from_archive(archive):
    num_DV = len(archive[0].candidate)
    num_Obj = len(archive[0].fitness)
    pf = np.zeros([len(archive),num_DV+num_Obj+1])
    for ii in range(len(archive)):
        pf[ii,0:num_Obj] = archive[ii].fitness
        pf[ii,num_Obj:num_Obj+num_DV] = archive[ii].candidate
        pf[ii,-1] = ii
    inds = np.argsort(pf[:,0],axis=0)
    PF = pf[inds[:],:]
    return PF


   
def nsga_replacement(random, population, parents, offspring, args):
    """Replaces population using the non-dominated sorting technique from NSGA-II.
    
    """
    survivors = []
    combined = population[:]
    combined.extend(offspring[:])
    
    # Perform the non-dominated sorting to determine the fronts.
    fronts = []
    pop = set(range(len(combined)))
    while len(pop) > 0:
        front = []
        for p in pop:
            dominated = False
            for q in pop:
                if combined[p] < combined[q]:
                    dominated = True
                    break
            if not dominated:
                front.append(p)
        fronts.append([dict(individual=combined[f], index=f) for f in front])
        pop = pop - set(front)
    
    # Go through each front and add all the elements until doing so
    # would put you above the population limit. At that point, fall
    # back to the crowding distance to determine who to put into the
    # next population. Individuals with higher crowding distances
    # (i.e., more distance between neighbors) are preferred.
    for i, front in enumerate(fronts):
        if len(survivors) + len(front) > len(population):
            # Determine the crowding distance.
            distance = [0 for _ in range(len(combined))]
            individuals = front[:]
            num_individuals = len(individuals)
            num_objectives = len(individuals[0]['individual'].fitness)
            for obj in range(num_objectives):
                individuals.sort(key=lambda x: x['individual'].fitness[obj])
                distance[individuals[0]['index']] = float('inf')
                distance[individuals[-1]['index']] = float('inf')
                for i in range(1, num_individuals-1):
                    distance[individuals[i]['index']] = (distance[individuals[i]['index']] + 
                                                         (individuals[i+1]['individual'].fitness[obj] - 
                                                          individuals[i-1]['individual'].fitness[obj]))
                
            crowd = [dict(dist=distance[f['index']], index=f['index']) for f in front]
            crowd.sort(key=lambda x: x['dist'], reverse=True)
            last_rank = [combined[c['index']] for c in crowd]
            r = 0
            num_added = 0
            num_left_to_add = len(population) - len(survivors)
            while r < len(last_rank) and num_added < num_left_to_add:
                if last_rank[r] not in survivors:
                    survivors.append(last_rank[r])
                    num_added += 1
                r += 1
            # If we've filled out our survivor list, then stop.
            # Otherwise, process the next front in the list.
            if len(survivors) == len(population):
                break
        else:
            for f in front:
                if f['individual'] not in survivors:
                    survivors.append(f['individual'])
    return survivors


def tournament_selection(random, population, args):
    """Return a tournament sampling of individuals from the population.
    
    Optional keyword arguments in args:
    
    - *num_selected* -- the number of individuals to be selected (default 1)
    - *tourn_size* -- the tournament size (default 2)
    
    """
    num_selected = args.setdefault('num_selected', 1)
    tourn_size = args.setdefault('tourn_size', 2)
    pop = list(population)
    selected = []
    for _ in range(num_selected):
        tourn = random.sample(pop, tourn_size)
        selected.append(max(tourn))
    return selected

def default_migration(random, population, args):
    """Do nothing.
    
    """
    return population

def default_observer(population, num_generations, num_evaluations, archive, args):
    """Do nothing."""    
    pass
    

def blend_crossover(random, candidates, args):
    """Return the offspring of blend crossover on the candidates.

    This function assumes that the candidate solutions are iterable
    and composed of values on which arithmetic operations are defined.
    It performs blend crossover, which is similar to a generalized 
    averaging of the candidate elements.

    .. Arguments:
       random -- the random number generator object
       candidates -- the candidate solutions
       args -- a dictionary of keyword arguments

    Optional keyword arguments in args:
    
    - *crossover_rate* -- the rate at which crossover is performed 
      (default 1.0)
    - *blx_alpha* -- the blending rate (default 0.1)
    - *lower_bound* -- the lower bounds of the chromosome elements (default 0)
    - *upper_bound* -- the upper bounds of the chromosome elements (default 1)
    
    The lower and upper bounds can either be single values, which will
    be applied to all elements of a chromosome, or lists of values of 
    the same length as the chromosome.
    
    """
    blx_alpha = args.setdefault('blx_alpha', 0.1)
    crossover_rate = args.setdefault('crossover_rate', 1.0)
    bounder = args['_ec'].bounder
        
    cand = list(candidates)
    if len(cand) % 2 == 1:
        cand = cand[:-1]
    random.shuffle(cand)
    moms = cand[::2]
    dads = cand[1::2]
    children = []
    for mom, dad in zip(moms, dads):
        if random.random() < crossover_rate:
            bro = []
            sis = []
            for index, (m, d) in enumerate(zip(mom, dad)):
                smallest = min(m, d)
                largest = max(m, d)
                delta = blx_alpha * (largest - smallest)
                bro_val = smallest - delta + random.random() * (largest - smallest + 2 * delta)
                sis_val = smallest - delta + random.random() * (largest - smallest + 2 * delta)
                bro.append(bro_val)
                sis.append(sis_val)
            bro = bounder(bro, args)
            sis = bounder(sis, args) 
            children.append(bro)
            children.append(sis)
        else:
            children.append(mom)
            children.append(dad)
    return children


    
def mutator(mutate): # This is a decorator function which is passed a mutator
    """Return an ecspy mutator function based on the given function.
   
    """
    def ecspy_mutator(random, candidates, args):
        mutants = list(candidates)
        for i, cs in enumerate(mutants):
            mutants[i] = mutate(random, cs, args)
        return mutants
    ecspy_mutator.__name__ = mutate.__name__
    ecspy_mutator.__dict__ = mutate.__dict__
    ecspy_mutator.__doc__ = mutate.__doc__
    ecspy_mutator.single_mutation = mutate
    return ecspy_mutator

@mutator    
def gaussian_mutation(random, candidate, args):
    """Return the mutants created by Gaussian mutation on the candidates.

    Optional keyword arguments in args:
    
    - *mutation_rate* -- the rate at which mutation is performed (default 0.1)
    - *mean* -- the mean used in the Gaussian function (default 0)
    - *stdev* -- the standard deviation used in the Gaussian function
      (default 1.0)
    
    """
    mut_rate = args.setdefault('mutation_rate', 0.1) # Sets the keyword argument if it hasn't been specified
    mean = args.setdefault('mean', 0.0)
    stdev = args.setdefault('stdev', 1.0)

    bounder = args['_ec'].bounder
    for i, c in enumerate(candidate):
        if random.random() < mut_rate:
            candidate[i] += random.gauss(mean, stdev)
    candidate = bounder(candidate, args)
    return candidate

    # Add strategy parameters here. A std of 1.0 for all mutations is bad!


class EvolutionaryComputation(object):
    """Represents a basic evolutionary computation.
    
    This class encapsulates the components of a generic evolutionary
    computation. These components are the selection mechanism, the
    variation operators, the replacement mechanism, the migration
    scheme, and the observers.
    
    Public Attributes:
    
    - *selector* -- the selection operator
    - *variator* -- the (possibly list of) variation operator(s)
    - *replacer* -- the replacement operator
    - *migrator* -- the migration operator
    - *archiver* -- the archival operator
    - *observer* -- the (possibly list of) observer(s)
    - *terminator* -- the (possibly list of) terminator(s)
    
    The following public attributes do not have legitimate values
    until after the ``evolve`` method executes:
    
    - *termination_cause* -- the name of the function causing 
      ``evolve`` to terminate
    - *generator* -- the generator function passed to ``evolve``
    - *evaluator* -- the evaluator function passed to ``evolve``
    - *bounder* -- the bounding function passed to ``evolve``
    - *maximize* -- Boolean stating use of maximization passed to ``evolve``
    - *archive* -- the archive of individuals
    - *population* -- the population of individuals
    - *num_evaluations* -- the number of fitness evaluations used
    - *num_generations* -- the number of generations processed
    - *logger* -- the logger to use (defaults to the logger 'ecspy.ec')
    
    Note that the attributes above are, in general, not intended to 
    be modified by the user. (They are intended for the user to query
    during or after the ``evolve`` method's execution.) However, 
    there may be instances where it is necessary to modify them 
    within other functions. This is possible but should be the 
    exception, rather than the rule.
    
    If logging is desired, the following basic code segment can be 
    used in the ``main`` or calling scope to accomplish that::
    
        logger = logging.getLogger('ecspy.ec')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler('ec.log', mode='w')
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    Protected Attributes:
    
    - *_random* -- the random number generator object
    - *_kwargs* -- the dictionary of keyword arguments initialized
      from the *args* parameter in the *evolve* method
    
    Public Methods:
    
    - ``evolve`` -- performs the evolution and returns the final
      archive of individuals
    
    """
    def __init__(self, random, log_file='hypervolume_log'):
        self.selector = tournament_selection
        self.variator = [blend_crossover, gaussian_mutation]
        self.replacer = nsga_replacement
        self.migrator = default_migration
        self.observer = default_observer
        self.archiver = best_archiver_mfuq
        self.terminator = generation_termination
        self.termination_cause = None
        self.generator = None
        self.evaluator = None
        self.bounder = None
        self.maximize = True
        self.archive = None
        self.population = None
        self.num_evaluations = 0
        self.num_generations = 0
        self.logger = logging.getLogger('ecspy.ec')
        self.logfile = log_file
        self._random = random
        self._kwargs = dict()
        
    def _should_terminate(self, pop, ng, ne):
        terminate = False
        fname = ''
        try:
            for clause in self.terminator:
                self.logger.debug('termination test using %s at generation %d and evaluation %d' % (clause.__name__, ng, ne))
                terminate = terminate or clause(population=pop, num_generations=ng, num_evaluations=ne, args=self._kwargs)
                if terminate:
                    fname = clause.__name__
                    break
        except TypeError:
            self.logger.debug('termination test using %s at generation %d and evaluation %d' % (self.terminator.__name__, ng, ne))
            terminate = self.terminator(population=pop, num_generations=ng, num_evaluations=ne, args=self._kwargs)
            fname = self.terminator.__name__
        if terminate:
            self.termination_cause = fname
            self.logger.debug('termination from %s at generation %d and evaluation %d' % (self.termination_cause, ng, ne))
        return terminate
    
    def evolve(self, generator, evaluator, pop_size=100, seeds=[], maximize=True, bounder=Bounder(), **args):
        """Perform the evolution.
        
        This function creates a population and then runs it through a series
        of evolutionary epochs until the terminator is satisfied. The general
        outline of an epoch is selection, variation, evaluation, replacement,
        migration, archival, and observation. The function returns a list of
        elements of type ``Individual`` representing the individuals contained
        in the final population.
        
        Arguments:
        
        - *generator* -- the function to be used to generate candidate solutions 
        - *evaluator* -- the function to be used to evaluate candidate solutions
        - *pop_size* -- the number of Individuals in the population (default 100)
        - *seeds* -- an iterable collection of candidate solutions to include
          in the initial population (default [])
        - *maximize* -- Boolean value stating use of maximization (default True)
        - *bounder* -- a function used to bound candidate solutions (default Bounder())
        - *args* -- a dictionary of keyword arguments

        Note that the *_kwargs* class variable will be initialized to the args 
        parameter here. It will also be modified to include the following 'built-in' 
        keyword arguments:
        
        - *_ec* -- the evolutionary computation (this object)
        
        """
        self._kwargs = args
        self._kwargs['_ec'] = self
        self.observer = self._kwargs.setdefault('observer',default_observer)

        self.termination_cause = None
        self.generator = generator
        self.evaluator = evaluator
        self.bounder = bounder
        self.maximize = maximize
        self.population = []
        self.archive = []

        bPlot = False
        if bPlot:
            plt.figure()
            plt.ion()
            plt.show()
       
        # Create the initial population.
        try:
            iter(seeds)
        except TypeError:
            seeds = [seeds]
        initial_cs = list(seeds)

        num_generated = max(pop_size - len(seeds), 0)
        i = 0
        while i < num_generated:
            cs = generator(random=self._random, args=self._kwargs)
            if cs not in initial_cs:
                initial_cs.append(cs)
                i += 1

        # Get the fidelity level before evaluation
        self.current_fidelity = self.observer(population=[], num_generations=0, num_evaluations=0, archive=self.archive, args=self._kwargs)

        initial_fit = evaluator(candidates=initial_cs, args=self._kwargs) 

        for cs, fit in zip(initial_cs, initial_fit):
            ind = Individual(cs, maximize=maximize)
            ind.fitness = fit
            self.population.append(ind)
        
        self.num_evaluations = len(initial_fit)
        self.num_generations = 0
        
        self.archive = self.archiver(random=self._random, population=list(self.population), archive=list(self.archive), args=self._kwargs)
             
        #################
        # MAIN LOOP 
        #################
        while not self._should_terminate(list(self.population), self.num_generations, self.num_evaluations):
            
            self.num_generations += 1
            print 'Generation: ', self.num_generations

            p1, p2 = [], []
            for ind in self.population:  p1.append(ind.candidate[0]); p2.append(ind.candidate[1])

            # Select individuals to be parents based on tournament selection (compare two random individuals and pick best)
            # Note with this random selection there can be repeats. 
            parents = self.selector(random=self._random, population=list(self.population), args=self._kwargs)
            parent_cs = [copy.deepcopy(i.candidate) for i in parents]
            offspring_cs = parent_cs

            s1, s2 = [], []
            for ind in parents:  s1.append(ind.candidate[0]); s2.append(ind.candidate[1])


            # Temperature decreasing encouraging less exploring as time goes on.
            T = 1 - float(self.num_generations) / (self._kwargs['max_generations']+1)
            # Crossover parents - r(1 - float(self.num_generations) / (self._kwargs['max_generations']+1) )**2eplaced general variator command with specific crossover and mutation
            self._kwargs['crossover_rate'] = 0.75 + 0.25*T**0.5
            offspring_cs = blend_crossover(random=self._random, candidates=offspring_cs, args=self._kwargs)
            
            c1, c2 = [], []
            for cs in offspring_cs: off = Individual(cs, maximize=maximize); c1.append(off.candidate[0]); c2.append(off.candidate[1])

            # Mutate blended parents to create offspring
            self._kwargs['mutation_rate'] = 0.25 - 0.15*T
            self._kwargs['stdev'] = 0.5 + 0.5*T
            #print self._kwargs['stdev']
            offspring_cs = gaussian_mutation(random=self._random, candidates=offspring_cs, args=self._kwargs)

            # Evaluate offspring.
            offspring_fit = evaluator(candidates=offspring_cs, args=self._kwargs)
            offspring = []
            for cs, fit in zip(offspring_cs, offspring_fit):
                off = Individual(cs, maximize=maximize)
                off.fitness = fit
                offspring.append(off)
            self.num_evaluations += len(offspring_fit)        

            m1, m2 = [], []
            for ind in offspring:   m1.append(ind.candidate[0]); m2.append(ind.candidate[1])

            if bPlot:
                plt.clf()
                plt.scatter(p1,p2,c='b',lw=0,alpha=0.5,label='Replaced Population')
                #plt.scatter(s1,s2,c='y',lw=0,alpha=0.5,label='Tournament Seletion')
                plt.scatter(c1,c2,c='r',lw=0,alpha=0.5,label='Blend Crossover')
                plt.scatter(m1,m2,c='g',lw=0,alpha=0.5,label='Gaussian Mutation')
                plt.xlim([0, 10])
                plt.ylim([0, 10])
                plt.legend()
                plt.title('Generation' + str(int(self.num_generations)))
                plt.draw()
                #time.sleep(0.01)

            # Archive individuals.
            self.archive = self.archiver(random=self._random, archive=list(self.archive), population=list(offspring), args=self._kwargs)
            
            bDebug = False
            if bDebug:
                print len(self.archive)
                PF = PF_from_archive(self.archive)
                plt.scatter(PF[:,0], PF[:,1], color='blue', label='Optimization 1')
                plt.legend()
                plt.grid(True)
                plt.show()

            # Replace individuals (using nsga-II non-dominated sorting)
            self.population = self.replacer(random=self._random, population=list(self.population), parents=parents, offspring=offspring, args=self._kwargs)

            # Observe fidelity level before new generation
            fidelity = self.observer(population=list(self.population), num_generations=self.num_generations, num_evaluations=self.num_evaluations, archive=self.archive, args=self._kwargs)
            
            # If changed fidelity level, re-evaluate archive
            if fidelity != self.current_fidelity:
                self.current_fidelity = fidelity

                # Re-evaluate current archive with new model fidelity
                if len(self.archive) > pop_size:    self.archive = reduce_archive(self.archive,pop_size)
                
                archive_cs = [copy.deepcopy(ind.candidate) for ind in self.archive]
                archive_fit = evaluator(candidates=archive_cs, args=self._kwargs)
                old_archive = []
                for cs, fit in zip(archive_cs, archive_fit):
                    ind = Individual(cs, maximize=maximize)
                    ind.fitness = fit
                    old_archive.append(ind)
                self.archive = []
                self.archive = self.archiver(random=self._random, archive=list(self.archive), population=list(old_archive), args=self._kwargs)
                self.num_evaluations += len(archive_fit)
                print 'Changed Fidelity'
                # Continue to evaluate next generation with new fidelity level

        if bPlot:
            plt.ioff()
            plt.close()
        return self.population

class NSGA2(EvolutionaryComputation):
    """Evolutionary computation representing the nondominated sorting genetic algorithm.
    
    This class represents the nondominated sorting genetic algorithm (NSGA-II)
    of Kalyanmoy Deb et al. It uses nondominated sorting with crowding for 
    replacement, binary tournament selection to produce *population size*
    children, and a Pareto archival strategy. The remaining operators take 
    on the typical default values but they may be specified by the designer.
    
    """
    def __init__(self, random=None):
        if random == None: 
            random = Random() 
            random.seed(time.time())
        EvolutionaryComputation.__init__(self, random)
    
    def evolve(self, generator, evaluator, pop_size=100, seeds=[], maximize=True, bounder=Bounder(), **args):
        args.setdefault('num_selected', pop_size)
        args.setdefault('tourn_size', 2)
        return EvolutionaryComputation.evolve(self, generator, evaluator, pop_size, seeds, maximize, bounder, **args)
