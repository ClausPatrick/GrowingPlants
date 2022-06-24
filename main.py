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
        data = {}   
        for i, cell in self.cell_list.items():
            if cell.fertile:                
                surround_per_cell = cell.position + w.surround_vector # vector of surround positions (8,1)
                limits = (0, w.size[0])
                pruned = np.delete(surround_per_cell, np.where(surround_per_cell.T==limits[0]), axis=0)
                pruned = np.delete(pruned, np.where(pruned.T==limits[1]), axis=0)
                data['cell_index'] = cell.index
                data['cell_position'] = cell.position
                data['other_root'] = w.grit['root'][tuple(pruned.T)]
                data['other_cell'] = w.grit['cell'][tuple(pruned.T)]
                data['other_position'] = pruned
        return data

    @property
    def choose_cell_position(self):
        root_vector = self.surrounding_data['other_root']        
        mask = np.where(root_vector == self.index, 0, 1)
        new_data = {}
        if np.any(mask):
            choices = len(mask)
            prob_abs = np.ones(choices) * mask
            prob_vec = prob_abs / np.sum(prob_abs)
            step_choice = np.random.choice(choices, p=prob_vec)
            data = self.surrounding_data
            new_data['cell_index'] = data['cell_index']
            new_data['cell_position'] = tuple(data['cell_position'])
            new_data['other_root'] = data['other_root'][step_choice]
            new_data['other_cell'] = data['other_cell'][step_choice]
            new_data['other_position'] = tuple(data['other_position'][step_choice])
            #print(f'choose_cell_position:: new_data: \n{new_data}')
            return new_data
        else:
            #print(f'No where to go!')
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
            #print(f'grow:: data: {data}')
            if not len(data):
                #print(f'grow:: growth inhibited. Pos: {self.position}, ix: {self.index}\n')
                #print(f'grow:: data: {data}\n')
                return
            c = Cell(data, self)
            next_step = data['other_position']             
            #print(f'grow:: Pos: {self.position}, ix: {self.index}')
            #print(f'grow:: next_step: {next_step}')
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
        '''Arg data should be dict'''
        Cell._counter = Cell._counter +  1
        self.role = 'cell'
        self.index = Cell._counter
        self.food = 1
        self.alive = True
        self.parent = 0
        self.dynasty = set()
        self.dynasty.add(self.index)
        self.old_dynasty = set()
        #self.lineage = [] # Contains alive values for all cells in dynasty
        self.age = 1
        self.root = root
        self.root_assumption = False
        self.root_connection = True
        self.state = 4
        self._state = 1.
        self.fertile = False
        self.victim_list = []
        self.data = {}
        self.data['friendly_neighbours'] = set()
        
        
        if len(data): # Second+ generation:
            self.position = data['other_position']
            other_cell_index = data['cell_index']
            other_root_index = data['other_root']
            #print(f'other_cell_index:{other_cell_index}, other_root_index:{other_root_index}')
            assert other_cell_index in root.cell_list, f"Cell__init__:: Error: i:{self.index}, data: {data}, \n{root.cell_list}"
            self.parent = root.cell_list[data['cell_index']]
            #print(f'Cell__init__:: Error: {e}, i:{self.index}, data: {data}') 
                
                
            self.parent_index = data['cell_index']
            self.parent_position = data['cell_position']
            #print(f'Cell:: {self.index}, parent: {self.parent_index}, {self.parent.dynasty}')
            self.dynasty.update(self.parent.dynasty) # Merge all parents ancestors.
            self.parent.food = self.parent.food / 2
            self.current_occupier = (data['other_root'], data['other_cell'])
            if self.current_occupier[0]:
                #print(f'current occupier: {self.current_occupier}, data: \n{data}')
                victim = w.population_dict[self.current_occupier[0]]
                vicitm_cell = victim.cell_list[self.current_occupier[1]]
                self.consume(victim, vicitm_cell)
                
        if not len(data): # First of a kind!
            self.position = root.position
            self.parent = root
            self.parent_index = root.index
            self.parent_position = root.position
            self.root_assumption = True
            
    def consume(self, victim, vicitm_cell):
        self.food = (self.food + 1) ** np.max((2, vicitm_cell.food + 1))
        vicitm_cell.alive = 0
        victim.cost = (victim.cost+1) * 2
        self.victim_list.append((victim.index, vicitm_cell.index))
        
    def tick(self):
        self.age = self.age + 1
        self.check_phase()
        try:
            self.food = np.max((0, w.grit['food'][tuple(self.position)] * np.min((2 - (self.age / MATURITY_AGE)), 0)))
        except Exception as e:
            raise ValueError(f"pos:{self.position}, age:{self.age}, wgfp:{w.grit['food'][tuple(self.position)]}")
        if self.state < 4 and self.state >= 3:
            self.fertile = True
        if self.state < 3:
            self.fertile = False
            
        if not self.lineage: # Test for Lineage. 
            #print(f'Lost access to ROOT! cell: {self.index}, parent: {self.parent.index}, {self.dynasty}')
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
        self.check_neighbours()
        
    def check_neighbours(self):
        data = {}
        fn = set()
        surround_per_cell = self.position + w.surround_vector # vector of surround positions (8,1)
        limits = (0, w.size[0]) # During pruning only positions within grit are passed thorugh.
        pruned = np.delete(surround_per_cell, np.where(surround_per_cell.T==limits[0]), axis=0) 
        pruned = np.delete(pruned, np.where(pruned.T==limits[1]), axis=0)
        o_r = w.grit['root'][tuple(pruned.T)]
        o_c = w.grit['cell'][tuple(pruned.T)]
        data['cell_index'] = self.index
        data['cell_position'] = self.position
        data['other_root'] = o_r
        data['other_cell'] = o_c
        data['other_position'] = pruned
        
        fn.add(((self.position), (self.root.index, self.index, self.state, self.age, self.root_connection, self.root_assumption)))
        prune_mask = np.where(np.array(o_r) != self.root.index)
        pruned_friendlies_cells = tuple(np.delete(o_c, prune_mask))
        pruned_friendlies_roots = tuple(np.delete(o_r, prune_mask))
        pruned_friendlies_pos = tuple(np.delete(pruned, prune_mask, axis=0))
        for i in zip(pruned_friendlies_pos, pruned_friendlies_roots, pruned_friendlies_cells):
            other_state = self.root.cell_list[i[2]].state
            other_age = self.root.cell_list[i[2]].age
            other_connection = self.root.cell_list[i[2]].root_connection
            other_assumption = self.root.cell_list[i[2]].root_assumption
            fn.add((tuple(i[0]),(i[1], i[2], other_state, other_age, other_connection, other_assumption)))    
        
        data['friendly_neighbours'] = fn
        #print(data['friendly_neighbours'])
        self.surround_data = data
        neighbours_neighbours = set()
        for n in pruned_friendlies_cells:
            neighbours_neighbours.update(self.root.cell_list[n].data['friendly_neighbours'])
        data['friendly_neighbours'].update(neighbours_neighbours)
        #print(f'ri:{self.root.index}, i:{self.index}, p:{self.position}, d:\n{data["friendly_neighbours"]}')
        for i in data['friendly_neighbours']:
            print(f'fn:\n{i}')
        print()
        
    @property
    def lineage(self):
        lineagelist = []
        for o_c in self.dynasty:
            lineagelist.append(self.root.cell_list[o_c].alive)
        self.root_connection = all(lineagelist)
        return self.root_connection
        
w = World(size=(32, 32))
w.spawn_root()
w.spawn_root()
    
