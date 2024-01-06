import random
import pandas as pd

# define helper functions
def count(seq):
    """Count the number of items in sequence that are interpreted as true."""
    return sum(map(bool, seq))


def first(iterable, default=None):
    """Return the first element of an iterable; or default."""
    return next(iter(iterable), default)


def is_in(elt, seq):
    """Similar to (elt in seq), but compares with 'is', not '=='."""
    return any(x is elt for x in seq)

class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return is_in(state, self.goal)
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError
    

# base class for formulating CSP problems (AIMA)
class CSP(Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases (for example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
    explicit representation). In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2], assignment)

        return sum(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if var in current and self.nconflicts(var, current[var], current) > 0]

### custom class for flight scheduling
class FlightSchedulerCSP(CSP):
    def __init__(self, flights_df, disruption_level=30, num_runways=1, neighbor_window=5, time_slot=5):
        """Initialize the flight scheduler CSP model."""
        self.diverted_slot = pd.Timestamp('2100-01-01')
        self.flights_df = flights_df
        self.disruption_level = disruption_level
        self.num_runways = num_runways
        self.neighbor_window = neighbor_window
        self.time_slot = time_slot

        variables = self.create_variables()
        domains = self.create_domains()
        neighbors = self.create_neighbors()
        constraints = self.create_constraints()

        CSP.__init__(self, variables, domains, neighbors, constraints)

    def create_variables(self):
        """Create variables for the CSP (each flight is a variable)."""
        return self.flights_df['code'].tolist()

    def create_domains(self):
        """Create domains for each variable based on flight schedule and disruption level."""

        # get the latest time in the dataset
        latest_time = self.flights_df['time_sch'].max()

        # calculate the last possible time slot considering the disruption
        last_time_slot = latest_time + pd.Timedelta(minutes=self.disruption_level) + pd.Timedelta(minutes=self.time_slot)

        # generate a list of all X-minute time slots, extending beyond the latest time in the dataset if necessary
        extended_time_slots = pd.date_range(
            start=self.flights_df['time_sch'].min(), end=last_time_slot, freq=f'{self.time_slot}T').tolist()

        domains = {}
        for index, row in self.flights_df.iterrows():
            flight_start_time = row['time_sch'] + pd.Timedelta(minutes=self.disruption_level)
            flight_time_slots = [time for time in extended_time_slots if time >= flight_start_time]
            domains[row['code']] = flight_time_slots

        for flight in domains:
            domains[flight].append(self.diverted_slot)

        return domains

    def create_neighbors(self):
        """Create neighbors for each variable. Returns all flights"""
        def find_neighbors(flight_code, flights_df, time_window=pd.Timedelta(minutes=self.neighbor_window)):
            flight_time = flights_df[flights_df['code'] == flight_code]['time_sch'].iloc[0]
            time_window_start = flight_time - time_window
            time_window_end = flight_time + time_window

            # find flights within the time window (excluding the flight itself)
            neighbor_flights = flights_df[
                (flights_df['time_sch'] >= time_window_start) & 
                (flights_df['time_sch'] <= time_window_end) &
                (flights_df['code'] != flight_code)
            ]

            return neighbor_flights['code'].tolist()

        variables = self.create_variables()  # Make sure this is called before using 'variables'
        return {flight: find_neighbors(flight, self.flights_df) for flight in variables}

    def create_constraints(self):
        """Define the constraint function."""
        def constraints(A, a, B, b, assignment):

            if a == self.diverted_slot or b == self.diverted_slot:
                return True  # always allow diverted slot

            # assuming runway capacity as before
            runway_capacity = self.num_runways # 3 ** (self.num_runways - 1)
            
            # Time buffer in minutes
            # time_buffer = pd.Timedelta(minutes=self.time_window)

            # check if A and B are scheduled too close to each other
            # if abs(a - b) < time_buffer and A != B:
            if a == b:
                # count flights at both time slots
                flights_at_a = sum(1 for flight, time in assignment.items() if time == a)
                flights_at_b = sum(1 for flight, time in assignment.items() if time == b)
                
                # check against runway capacity
                if flights_at_a >= runway_capacity or flights_at_b >= runway_capacity:
                    return False

            return True

        return constraints


# Min-conflicts Hill Climbing search for CSPs
def min_conflicts(csp, max_steps=100_000, resolve_strategy="early"):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts.
    Parameters:
    - csp: The CSP instance.
    - max_steps: Maximum number of steps for the algorithm.
    - resolve_strategy: Strategy to resolve conflicts ('early', 'force', 'divert').
    """
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)

    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        # if i % 100 == 0: print(f"Step: {i:,}")
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current

        # Safeguard against getting stuck at local minima
        if resolve_strategy == "early":
            if len(current) == len(csp.variables): 
                return current
        elif resolve_strategy == "force" and i > 500:
            for flight in conflicted:
                # choose the minimum available time slot
                current[flight] = min(csp.domains[flight])
                csp.assign(flight, current[flight], current)
            return current
        elif resolve_strategy == "divert" and i > 500:
            for flight in conflicted:
                # assign to the diverted slot
                csp.assign(flight, csp.diverted_slot, current)
            return current

        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)

    return None

def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose the earliest time."""
    # sort the domain values by time, then by the number of conflicts
    domain_sorted_by_time_and_conflicts = sorted(
        csp.domains[var],
        key=lambda val: (csp.nconflicts(var, val, current), val)
    )
    # now select the value that minimizes conflicts and is the earliest in time
    if domain_sorted_by_time_and_conflicts:
        return domain_sorted_by_time_and_conflicts[0]  # the first element captures this info
    return csp.diverted_slot

def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose the earliest time."""
    # print(f"var: {var} \ndomains: {csp.domains[var]} \nconflicts: {csp.nconflicts(var, csp.domains[var], current)}")
    return argmin_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))


identity = lambda x: x


def argmin_tie(seq, key=identity):
    """Return a minimum element of seq; break ties at random."""
    return min(seq, key=key)
