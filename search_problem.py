import pandas as pd 
import numpy as np
import itertools
# _________________________________
# search problem
class reschedule_problem:

    def __init__(self,df, n_runway = 1, disruption_dur = 60,\
                timeslot_dur = 5, max_delay = 120):
        # customized attributes of the problem
        self.df = df
        self.move = df['code']
        self.n_runway = n_runway
        self.resumetime = self.df.iloc[0,0] + pd.Timedelta(minutes = disruption_dur)
        self.hour = self.resumetime.hour
        self.min = self.resumetime.minute
        self.time_slot_duration = timeslot_dur
        self.max_delay = max_delay
        # parsing the attribute for more useful format
        self.date = self.df.loc[0,'time_sch'].day
        self.month = self.df.loc[0,'time_sch'].month
        self.year = self.df.loc[0,'time_sch'].year
        self.initial = pd.DataFrame(columns = ["time_new","util"], index= []) # to store the solution
        self.solution = None

    def actions(self, state):
        """return a list of n_runway flight(s) that has/have is expected to land
        and is not assigned a time slot under the state"""
        # get the list of unassigned flight where the time_sch is (over)due under current time slot 
        assigned_flight = list([flight for flight in state.index])
        unassigned_flight = self.df.query("code not in @assigned_flight").copy()
        # subset the flights that are not in time yet
        year, month, date, hour, min = self.parse_state_time(state)
        current_time = pd.to_datetime(f"{year} {month} {date} {hour}:{min}")
        min_time = pd.to_datetime(f"{year} {month} {date} {hour}:{min}") - pd.Timedelta(minutes = self.max_delay)
        unassigned_flight['divert'] = unassigned_flight.apply(lambda flight: flight['time_sch'] < min_time,axis = 1)
        unassigned_flight['to_sch'] = unassigned_flight.apply(lambda flight: flight['time_sch'] <=current_time  ,axis = 1)


        # return the pd.Series of unassigned flight that has passed its schedule time
        flight_to_assign = unassigned_flight[unassigned_flight['to_sch'] & ~(unassigned_flight['divert'])]['code']
        # return the combination of flights (namely when n_runway > 1)
        flight_to_assign = [comb for n in range(1, self.n_runway+1) for comb in itertools.combinations(flight_to_assign, n)]
        # flight to divert
        flight_diverted = unassigned_flight[unassigned_flight['divert']]['code']
        
        return flight_to_assign, flight_diverted # pd.Series
        
    def parse_state_time(self,state):
        """Use to get the value in the schedule and 
        parse the time of the next time slot"""
        if len(state) <=0: 
            year = self.year
            month = self.month
            date = self.date
            hour = self.hour
            min = self.min

        else:
            timestr = state['time_new'].iloc[-1] # get the time_sch
            year, month, date, time = timestr.split(" ")
            hour , min = time.split(":")
            # get the current time
            min = int(min)
            min += self.time_slot_duration
            if min // 60 <= 1:
                hour = int(hour) + min // 60
                min = min % 60
                if hour // 24 <= 1:
                    date = int(date) + hour // 24
                    hour = hour % 24
                
        return year, month, date, hour,min
    
    def result(self, state, flight_diverted:list, flights:list):
        """return the state in a form of dictionary that is the result of a given move"""
        # parse the time data
        year, month, date, hour, min = self.parse_state_time(state)
        for fl in flights:
            time_sch = f"{year} {month} {date} {hour:02d}:{min:02d}"
            # get the utility
            util = self.compute_util(fl, year, month, date, hour, min)
            # print(f"Scheudling {flights} for {time_sch}")
            state.loc[fl] = [time_sch, util]
    
        # add the diverted flight into the state
        for fl in flight_diverted:
            util = self.compute_util(fl,1970, 0,0,0,0)
            state.loc[fl] = ["1970 01 01 00:00", util]
            # print(f"Diverting {flights}")

        state = state.sort_values("time_new")

        return state # pd.DataFrame
    
    def compute_util(self, flcode, year, month, date, hour, min):
        """Compute the utility of a given rescheduled flight. This is defined
        as the time difference between the original scheduled time and the 
        new scheduled time"""
        if flcode is None:
            return 0 
        elif year != 1970:
            time_sch_org = self.df.query("code == @flcode")['time_sch'] # type pd Series
            time_sch_new = pd.to_datetime(f"{year} {month} {date} {hour}:{min}")
            # compute the time delayed
            delay = time_sch_org - time_sch_new
            delay = delay.reset_index(drop = True)[0].total_seconds() / 60
            return delay
        elif year == 1970:
            # compute utility of diverted flight
            delay = - self.max_delay 
            return delay
    
    def utility(self, state):
        """Compute aggregate utility of a given state"""
        agg_util = state['util'].sum()
        return agg_util
        
    def goal_test(self, state):
        "return True if the state is terminal"
        flight_assigned = [flight for flight in state.index] 
        if len(flight_assigned) == len(self.df):
        # !!! this is not a robust way since time slot can be none
            return True
        
    def solve(self, solver_algo):
        self.solution = solver_algo(self)    
        
    def display(self):
        """ Display the self.solution that is of type pandas dataframe
        to compare the solution with the set up that we try to solve 
        """
        if self.solution is None:
            print("The problem has not been solved. Pass a solving algorithm to the .solve method")
            raise NotImplementedError
        elif type(self.solution) != pd.core.frame.DataFrame:
            print("""THe solution returned by the algorithm cannot be parsed because it is 
                  not of type pd.core.frame.DataFrame""")
        
        display_df = self.df.copy()
        display_df = pd.merge(self.df[['code','time_sch','pass_load']], self.solution, left_on= "code", right_index = True)
        display_df['time_new'] = pd.to_datetime(display_df['time_new'])
        display_df['time_dff'] = display_df.apply(lambda x: (x['time_sch'] - x['time_new']).total_seconds()/ 60,axis = 1 )
        display_df['time_dff'] = display_df['time_dff'].apply(lambda x: -x if x <= 0 else "diverted" )
        return display_df


# __________________________
# search problem with utility as a function of time delayed and passenger load
class reschedule_updated_u(reschedule_problem):

    def __init__(self, df, n_runway = 1, disruption_dur = 60,
                timeslot_dur = 5, max_delay = 120):
        super().__init__(df, n_runway , disruption_dur,
                timeslot_dur, max_delay)


    def compute_util(self, flcode, year, month, date, hour, min):
        """Compute the utility of a given rescheduled flight. This is defined
        as the time difference between the original scheduled time and the 
        new scheduled time"""
        if flcode is None:
            return 0 
        elif year != 1970:
            time_sch_org = self.df.query("code == @flcode")['time_sch'] # type pd Series
            time_sch_new = pd.to_datetime(f"{year} {month} {date} {hour}:{min}")
            # compute the time delayed
            delay = time_sch_org - time_sch_new
            delay = delay.reset_index(drop = True)[0].total_seconds() / 60

        elif year == 1970:
            # compute utility of diverted flight
            delay = - self.max_delay

        pass_load = self.df.query("code == @flcode")['pass_load'].values
        util = delay * np.arctan(pass_load/200) /np.pi * 2

        return util[0]

# __________________________
# search problem with alternative u function
class reschedule_custom_u(reschedule_problem):

    def __init__(self, df,util_f, n_runway = 1, disruption_dur = 60,
                timeslot_dur = 5, max_delay = 120,):
        super().__init__(df, n_runway , disruption_dur,
                timeslot_dur, max_delay)
        self.util_f = util_f


    def compute_util(self, flcode, year, month, date, hour, min):
        """Compute the utility of a given rescheduled flight. This is defined
        as the time difference between the original scheduled time and the 
        new scheduled time"""
        if flcode is None:
            return 0 
        elif year != 1970:
            time_sch_org = self.df.query("code == @flcode")['time_sch'] # type pd Series
            time_sch_new = pd.to_datetime(f"{year} {month} {date} {hour}:{min}")
            # compute the time delayed
            delay = time_sch_org - time_sch_new
            delay = delay.reset_index(drop = True)[0].total_seconds() / 60

        elif year == 1970:
            # compute utility of diverted flight
            delay = - self.max_delay

        pass_load = self.df.query("code == @flcode")['pass_load'].values
        util = self.util_f(delay, pass_load)

        return util[0]


# ____________________________
# node
class Node:
    """This refenece to the anima class node"""
    def __init__(self, state,parent = None, action = None, path_cost = 0):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0 
        if parent:
            self.depth = parent.depth + 1
    def __repr__(self):
        """define printing output of the class type"""
        return "<Node of depth {}>".format(self.depth)
    
    def __lt__(self, node):
        """Compare the depth of the nodes -- for use in 
        sorting the *frontier*"""
        return self.depth < node.depth
    
    def expand(self, problem):
        """reach the nodes reachable"""
        flight_to_assign, flight_diverted = problem.actions(self.state)
        return [self.child_node(problem, action, flight_diverted) 
                  for action in flight_to_assign]

    def child_node(self, problem,action, flight_diverted):
        """ Return a Node object representing the child node        
        """
        parent_state = self.state.copy()
        next_state = problem.result(parent_state, flight_diverted, action)
        # print(next_state)
        next_node = Node(next_state, parent=self, action = action, path_cost = problem.utility(next_state))
        return next_node # node object

    
# ____________________________________________
# bfs graph
def best_first_graph_search(problem):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    # adding first node
    node = Node(problem.initial)
    print(f"The airport resumed service at {problem.hour}:{problem.min:02d}")
    iterations = 1
    # applying the goal test when generating the node
    if problem.goal_test(node.state):
        return(iterations, node)

    # expand the frontier based on the priority queue
    # the current best candidate for extension
    frontier = list()
    frontier.append(tuple([node.path_cost,node ]))

    while frontier:
        # get the next node in the frontier
        node = frontier.pop()[1]
        # applying the goal test when expanding the node
        if problem.goal_test(node.state):
            return node.state
        # for every child in the frontier
        for child in node.expand(problem): # child is a node
            frontier.append(tuple([int(child.path_cost),child]))
            frontier.sort(reverse = True)
        
    return None # otherwise return node.state
    
# ____________________________________________
# bfs graph
def breadth_first_search(problem):
    """Search the nodes with the lowest depth scores first.
    """
    # adding first node
    node = Node(problem.initial)
    print(f"The airport resumed service at {problem.hour}:{problem.min:02d}")
    iterations = 1
    # applying the goal test when generating the node
    if problem.goal_test(node.state):
        return(iterations, node)

    solutions = []
    # expand the frontier based on the priority queue
    # the current best candidate for extension
    frontier = list()
    frontier.append(tuple([node.depth,node ]))

    while frontier:
        # get the next node in the frontier
        node = frontier.pop()[1]
        # applying the goal test when expanding the node
        if problem.goal_test(node.state):
            # add the node to the list of dictionary
            solutions.append(tuple([node.path_cost,node]))
            print(f"A terminal state has been reached: total {len(solutions)} solutions")
        else:
            # for every child in the frontier
            for child in node.expand(problem): # child is a node
                print(f"Iteration: {iterations} at depth {child.depth}")
                print((f"Check if the depth is consistent withe the node state."))
                print(f"There should be {len(child.state)/problem.n_runway} timeslot iterated.")
                frontier.append(tuple([int(child.depth),child]))
                frontier.sort(reverse = True)
                print(f"frontier length: {len(frontier)}")
                iterations+= 1
            
                if iterations % 10000 == 0:
                    command_input = input("Continue or break?")
                    try:
                        command_input = command_input.lower.strip() 
                    except:
                        command_input = None
                    if command_input != "break":
                        break

    # <return the solution from the list that has the maximum utility>
    # compute the utilityof all the solution yielded
    solutions.sort(reverse = True)
    # return the   
    return solutions