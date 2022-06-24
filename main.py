import random
import numpy as np
import pandas as pd

#%%%%%%%%%%%%%%%%%%%%%%%%%% WORLD CLASS %%%%%%%%%%%%%%%%%%%%%%%%%%#
class World:
    surround_vector = np.array(((0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)))
    surround_org_vector  = np.array(((0, 0), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)))
    def __init__(self, size=(128, 128)):
        self.role = 'world'
        self.size = size
        self.population_dict = {}
        self.grit = {}
        self.grit['root'] = np.zeros(size).astype(int)
        self.grit['cell'] = np.zeros(size).astype(int)
        self.grit['food'] = np.ones(size).astype(int)    
        self.grit['cell_alive'] = np.zeros(size).astype(int)
        self.grit['root_alive'] = np.zeros(size).astype(int)
        template = np.hstack((np.arange(size[0]/2), np.arange(size[0]/2)[::-1]))
        v = ((template - size[0] / 2) / (size[0] ) + 1).reshape(1, -1)
        h = ((template - size[1] / 2) / (size[1] ) + 1).reshape(1, -1)
        self.grit['food'] = (v.T @ h) * 2
    
    def update(self, obj, pos, mode='initiate'):
        if mode == 'initiate':
            l = obj.alive
            i = obj.index
            r = obj.role
            self.grit[r+'_alive'][pos] = l
            try:
                self.grit[r][pos] = i
            except Exception as e:
                print(f'update_grit:: Error: {e}, {r}')
        else:
            print(f'W:: update: mode not parsed: {mode}')
        '''
        update_grit:: Error: name 'role' is not defined, cell
        update_grit:: Error: name 'role' is not defined, root
        '''        
    

    def spawn_root(self):
        index = len(self.population_dict) + 1
        self.population_dict[index] = Root(np.random.randint(w.size, size=2))
        
    def tick(self):
        for i, root in self.population_dict.items():
            root.tick()
            
    def __str__(self):
        s = ''
        for index, root in self.population_dict.items():
            s = s + f'Root {index}, food: {root.food}, age: {root.age}, cells: {root.live_cells}, live: {root.alive}\n'   
        return s

#%%%%%%%%%%%%%%%%%%%%%%%%%% ROOT CLASS %%%%%%%%%%%%%%%%%%%%%%%%%%#

class Root:
    _counter = 0
    def __init__(self, initial_position):
        self.role = 'root'
        self.position = tuple(initial_position)
        Root._counter = Root._counter + 1
        self.index = Root._counter
        w.grit['root'][tuple(initial_position)] = self.index
        w.population_dict[self.index] = self
        self.cell_list = {}
        self.alive = 1
        self.food = 100
        self.parent = None
        self.cost = 0
        self.total_cell_food = 0
        self.age = 0
        self.live_cells = 1
        self.state = 3
        
        
    def tick(self):
        if self.alive:
            self.metabolise()
            self.grow()
    
    @property
    def gather_surrounding(self):
        columns = ['cell_index', 'cell_position', 'other_root', 'other_cell', 'other_position']
        data = {}
        lookup = []        
        for i, cell in self.cell_list.items():
            if cell.fertile:                
                surround_per_cell = cell.position + w.surround_vector
                for j in surround_per_cell: #j: spot_surround_per_cell
                    if j[0] >= 0 and j[1] >= 0 and j[0] < w.size[0] and j[1] < w.size[1]:                    
                        j_tup = tuple(j)
                        if j_tup not in data:
                            cr = w.grit['root'][j_tup]
                            cl = w.grit['cell'][j_tup]
                            data[j_tup] = (cell.index, cell.position, cr, cl, j_tup)        
            #else:
                #print(f'Cell death: {cell.index}')
        surrounding_data = pd.DataFrame(columns=columns, data=list(data.values()))
        return surrounding_data

    @property
    def choose_cell_position(self):
        root_vector = self.surrounding_data['other_root']        
        mask = np.where(root_vector == self.index, 0, 1)
        if np.any(mask):
            choices = len(mask)
            prob_abs = np.ones(choices) * mask
            prob_vec = prob_abs / np.sum(prob_abs)
            step_choice = np.random.choice(choices, p=prob_vec)
            data = self.surrounding_data.iloc[step_choice]
            return data
        else:
            print(f'No where to go!')
            return []
            

    def metabolise(self):
        if self.food < 0: #or self.live_cells == 0:
            self.alive = 0
        else:
            self.age = self.age + 1
            total_cell_food = 0
            live_cells = 0
            for i, cell in self.cell_list.items():
                if cell.alive:
                    live_cells = live_cells + 1
                    cell.tick()
                    total_cell_food = total_cell_food + cell.food
            self.live_cells = live_cells
            self.total_cell_food = total_cell_food
            self.food = self.total_cell_food - self.cost
            self.cost = 0  
            
    def grow(self):
        self.cost = self.cost + 1
        self.surrounding_data = self.gather_surrounding
        if len(self.surrounding_data):
            data = self.choose_cell_position
            print(f'grow:: data: {data}')
            if not len(data):
                print(f'grow:: growth inhibited. Pos: {self.position}, ix: {self.index}\n')
                print(f'grow:: data: {data}\n')
                return
            c = Cell(data, self)
            next_step = data['other_position']             
            print(f'grow:: Pos: {self.position}, ix: {self.index}')
            print(f'grow:: next_step: {next_step}')
        else:
            #print(f'first cell on pos {self.position}')
            next_step = self.position
            c = Cell([], self)
        self.cell_list[c.index] = c
        w.update(c, tuple(next_step), mode='initiate')
        w.update(self, tuple(next_step), mode='initiate')
        #w.grit['cell'][tuple(next_step)] = c.index
        #w.grit['root'][tuple(next_step)] = self.index
        
    
    def __str__(self):
        s = f'Root:: {self.index}\n'
        for i, cell in self.cell_list.items():
            if cell.alive:
                s = s + f'Cell:: {cell.index}, food: {cell.food}, victims: {cell.victim_list}, state: {cell.state}, live:{cell.alive}\n'
        return s
        
#%%%%%%%%%%%%%%%%%%%%%%%%%% CELL CLASS %%%%%%%%%%%%%%%%%%%%%%%%%%#

FOOD_DEATH_THRESHOLD = 0.001
MATURITY_AGE = 20

class Cell:
    _counter = 0
    def __init__(self, data=None, root=None):
        Cell._counter = Cell._counter +  1
        self.role = 'cell'
        self.index = Cell._counter
        self.food = 1
        self.alive = 1
        self.parent = 0
        self.dynasty = set()
        self.dynasty.add(self.index)
        self.old_dynasty = set()
        #self.lineage = [] # Contains alive values for all cells in dynasty
        self.age = 1
        self.root = root
        self.state = 4
        self._state = 1
        self.fertile = False
        self.victim_list = []
        
        if len(data): # Second+ generation:
            self.position = data['other_position']
            self.parent = root.cell_list[data['cell_index']]
            self.parent_index = data['cell_index']
            self.parent_position = data['cell_position']
            #print(f'Cell:: {self.index}, parent: {self.parent_index}, {self.parent.dynasty}')
            self.dynasty.update(self.parent.dynasty) # Merge all parents ancestors.
            self.parent.food = self.parent.food / 2
            self.current_occupier = (data['other_root'], data['other_cell'])
            if self.current_occupier[0]:
                victim = w.population_dict[self.current_occupier[0]]
                vicitm_cell = victim.cell_list[self.current_occupier[1]]
                self.consume(victim, vicitm_cell)
                
        if not len(data): # First of a kind!
            self.position = root.position
            self.parent = root
            self.parent_index = root.index
            self.parent_position = root.position
            
    def consume(self, victim, vicitm_cell):
        self.food = (self.food + 1) ** np.max((2, vicitm_cell.food + 1))
        vicitm_cell.alive = 0
        victim.cost = (victim.cost+1) * 2
        self.victim_list.append((victim.index, vicitm_cell.index))
        
    def tick(self):
        self.age = self.age + 1
        self.check_phase()
        self.food = np.max((0, w.grit['food'][self.position] * np.min((2 - (self.age / MATURITY_AGE)), 0)))
        if self.state < 4 and self.state >= 3:
            self.fertile = True
        if self.state < 3:
            self.fertile = False
            
        if not self.lineage:
            #print(f'Lost access to ROOT! cell: {self.index}, parent: {self.parent}, {self.dynasty}')
            
            self.old_dynasty.update(self.dynasty)
            self.dynasty = set()
        self.check_neighbours()
            
    def check_phase(self):
        '''Phase depends on age, food, 1/lineage'''   
        lineage = len(self.old_dynasty) + len(self.dynasty)
        prob = self.food * (MATURITY_AGE / self.age) * (lineage / 5)
        prob_vec = ((np.min((1, np.max((0, prob)))), (1 - np.min((1, np.max((0, prob)))))))
        #print(f'check_phase:: cell:{self.index}, food: {self.food}, age: {self.age}, lineage: {lineage}')
        p = np.random.choice((-0.5, 1), p=prob_vec)
        self._state = self._state + p
        if self._state < 0:
            self._state = 1
            self.state = np.max((0, self.state - 1))
        #print(f'check_phase:: cell:{self.index}, st:{self.state}, s_:{self._state}')
        
    def root_elect(self):
        pass
        
    def check_neighbours(self):
        field_all_arr = self.position + w.surround_vector
        field_all = tuple((field_all_arr).T)

        field = np.delete(field_all, np.where(np.array(field_all) >= 31)[1], axis=1)
        field = np.delete(field_all, np.where(np.array(field_all) < 0)[1], axis=1)
        occupants = w.grit['root'][tuple(field)]
        data = np.vstack((field[0], field[1]))
        print(f'data: {data}\n')
        

        
    @property
    def lineage(self):
        lineagelist = []
        for o_c in self.dynasty:
            lineagelist.append(self.root.cell_list[o_c].alive)
        return all(lineagelist)
        
w = World(size=(32, 32))
w.spawn_root()
w.spawn_root()        
