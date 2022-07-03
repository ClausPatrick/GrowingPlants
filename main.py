#imports


#%%%%%%%%%%%%%%%%%%%%%%%%%% WORLD CLASS %%%%%%%%%%%%%%%%%%%%%%%%%%#
class World:
    surround_vector = np.array(((0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)))
    surround_org_vector  = np.array(((0, 0), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)))
    def __init__(self, size=(128, 128)):
        #size = (size[0]+1, size[1]+1)
        self.role = 'world'
        assert size[0] == size[1], 'Aspect ratio other than 1:1 not implemented yet.'
        self.size = size
        self.size_prod = size[0]*size[1]
        self.cell_list = {}
        self.population_dict = {}
        self._population_counter = 0
        self.grit = {}
        self.count_of_resources = 4
        self.grit['cell'] = np.zeros(size).astype(int)
        self.grit['resources'] = np.ones((size[0], size[1], self.count_of_resources)).astype(int)    
        self.grit['cell_alive'] = np.zeros(size).astype(int)
        self.grit['gene_id'] = {}
        self.gene_pool = np.zeros((self.size[0], self.size[1], self.size_prod)).astype(int)
        template = np.hstack((np.arange(size[0]/2), np.arange(size[0]/2)[::-1]))
        v = ((template - size[0] / 2) / (size[0] ) + 1).reshape(1, -1)
        h = ((template - size[1] / 2) / (size[1] ) + 1).reshape(1, -1)
        for r in range(self.count_of_resources):
            self.grit['resources'][:,:,r] = (v.T @ h) * 2
        self.extinction = False
        self.random_spots = np.random.permutation(self.size_prod).reshape(self.size)

    def tick(self):
        population_copy = self.population_dict.copy()
        for k, v in population_copy.items():
            if v.alive:
                v.tick()
                    
    def spawn(self, parent=None, role='cell', position=None):
        new_index = self._population_counter 
        if new_index > self.size_prod:
            print(f'World is overpopulated')
            return
        #print(f'spawn1:: posiition: \n{position}')
        if position == None and parent == None:
            hH = (w.size[0] // 3) * 2
            hL = (w.size[1] // 3)
            p = ((hH, hL), (hL, hH), (hH, hH), (hL, hL))
            position = p[new_index%4]
            parent = 0         
        if position == 'random':
            rand_pos = np.where(self.random_spots == self._population_counter)            
            position = (int(rand_pos[0]), int(rand_pos[1]))            
        self._population_counter = self._population_counter + 1
        self.population_dict[new_index] = Cell(parent=parent, position=position)

    def get_new_genes(self, obj):
        new_genes = np.random.randint(255, size=self.size_prod)
        pos = obj.position
        self.gene_pool[pos[0], pos[1], :] = new_genes
        return new_genes
        
        
    def get_genes(self, pos):
        #print(f'get_genes:: obj:{obj}, type:{type(obj)}')
        #logging.debug(f'Object {self.role}:: gene pos:{pos}.')
        genes = self.gene_pool[pos[0], pos[1], :]
        #print(f'get_genes:: genes:{genes}')
        return genes

    def set_genes(self, obj, genes):
        #print(f'set_genes:: obj:{obj}, type:{type(obj)}')
        if isinstance(obj, Cell):
            pos = obj.position
        elif isinstance(obj, (int, np.int32)):
            pos = self.cell_list[obj].position 
        else:
            raise TypeError(f'set_genes:: obj:{obj}, type:{type(obj)}')
        gene_len = len(genes)
        if gene_len < self.size_prod:
            genes = list(genes) + [0]*(self.size_prod-gene_len)
        else:
            genes = list(genes)
        self.gene_pool[pos[0], pos[1], :] = genes
        
        
    def activate(self, obj):
        obj_index = obj.index
        if isinstance(obj, Cell):
            self.cell_list[obj_index] = obj
            obj_position = obj.position
            self.grit['cell_alive'][obj_position] = 1
            self.grit['cell'][obj_position] = obj_index
            self.grit['gene_id'][obj.position] = obj.gene_id
            self.set_genes(obj, obj.gene_id)


            
GENE_SIZE = 8        
class Cell:
    _gene_blueprint = np.random.randint(255, size=GENE_SIZE)
    _counter = 0
    def __init__(self, parent=None, data=None, position=None, from_cell=False):
        '''Arg data should be dict'''
        Cell._counter = Cell._counter +  1
        self.role = 'cell'
        self.index = Cell._counter
        self.food = 1 #+ np.random.randint(100, size=(1))
        self.food_budged = (0,) * w.count_of_resources
        self.alive = True
        self.parent = 0                
        self.dynasty = set() # Contains all Cell indices this Cell instance originated from.
        self.dynasty.add(self.index) 
        self.old_dynasty = set() # Upon root connection loss dynasty dict will be purged into this.
        self.age = 1
        self.state = 4
        self._state = 1.  # Progress track keeper to shift through phases
        self.fertile = False # New cells can only grow from cells that are fertile
        self.victim_list = [] # Keep track of what gets eaten.
        self.data_surrounding = {} # Pertinent for new root election.
        #self.data_surrounding['friendly_neighbours'] = set()
        if not from_cell:
            assert position != None
            self.position = position
            self.gene_id = self._express_gene_id
            #self.gather_surrounding()
              
            
        if from_cell:
            if isinstance(parent, Cell):
                self.parent = parent
                self.dynasty.update(parent.dynasty)
                #self.gather_surrounding()
                    
        self.surround_per_cell = self.position + w.surround_vector
        limits = (0, w.size[0], w.size[1])
        pruned = self.surround_per_cell[:]
        pruned = np.delete(pruned, np.where(pruned<limits[0])[0], axis=0) # Dropping lower bounds, V&H
        pruned = np.delete(pruned, np.where(pruned[:, 0]>=limits[1])[0], axis=0)  # Dropping upper bounds, V
        self.surround_vector = np.delete(pruned, np.where(pruned[:, 1]>=limits[2])[0], axis=0)  # Dropping upper bounds, H

        logging.debug(f'Object {self.role} created at {position}.')
        w.cell_list[self.index] = self
        w.activate(self)
        
    def tick(self):
        self.gather_surrounding()
        sensory_dict = self.process_sensory
        peers_info = sensory_dict['peers_info_exchange']
        peers_food = sensory_dict['peers_food_exchange']
        potential_mates = sensory_dict['peers_mate_potential']
        potential_victims = sensory_dict['peers_eat_potential']
        self.sensory_dict = sensory_dict
        if len(peers_info):
            self.share_info(peers_info)
        if len(peers_food):
            self.share_info(peers_food)
        if len(potential_mates):
            self.mate(potential_mates)
        if len(potential_victims):
            self.eat(potential_victims)
            
              
    @property
    def get_peer_data(self):
        return  {'peers_info_exchange': len(self.sensory_dict['peers_info_exchange']), 
                'peers_food_exchange': len(self.sensory_dict['peers_food_exchange']), 
                 'peers_mate_potential': len(self.sensory_dict['peers_mate_potential']), 
                'peers_eat_potential': len(self.sensory_dict['peers_eat_potential'])}

    @property
    def process_sensory(self):
        sensory_dict = {}
        keys = ('peers_info_exchange',
                'peers_food_exchange',
                'peers_mate_potential',
                'peers_eat_potential')
        data_col_ix = (8, 9, 10, 11)
        data = np.array(list(self.data_surrounding.values()), dtype=object )
        for i, k in zip(data_col_ix, keys):
            info_ = data[np.where(data[:, i]==1)] # and data[:, 3]>0
            info = info_[:, 3][np.where(info_[:, 3]>0)] # and data[:, 3]>0
            sensory_dict[k] = info
        return sensory_dict
        
        
    def grow(self):        
        position = self.choose_position
        w.spawn(parent=self, position=position)
        logging.debug(f'Object {self.role}::{self.index} Grow at position:{position}.')
    
    def eat(self, peer_list):
        logging.debug(f'Object {self.role}::{self.index}.')
    
    def mate(self, peer_list):
        '''Choose one of mates in peer_list based on its features such as:
        corr, food, age'''
        print(f'mate::Object {self.role}::{self.index}:: surr:\n{self.data_surrounding}.\n')
        data = np.array(list(self.data_surrounding.values()), dtype=object )
        mate_data = data[np.where(data[:, 10])]
        print(f'mate::Object {self.role}::{self.index}:: mate_data:\n{mate_data}.\n')
        
    def choose_position(self):
        logging.debug(f'Object {self.role}::{self.index}.')       
        
    def share_info(self, peer_list):
        if isinstance(peer_list, (tuple, list, np.ndarray)) and len(peer_list):
            for c in peer_list:
                w.cell_list[c].data_surrounding.update(self.data_surrounding)
                self.data_surrounding.update(w.cell_list[c].data_surrounding) # Lastly, update ourself
        else:            
            raise TypeError
        
    def share_food(self, peer_list):
        if isinstance(peer_list, (tuple, list, np.ndarray)) and len(peer_list):           
            collected_food = self.food            
            for c in peer_list:
                collected_food = collected_food + w.cell_list[int(c)].food
            collected_food = collected_food / len(peer_list)            
            for c in peer_list:
                w.cell_list[int(c)].food = collected_food
        else:            
            raise TypeError
                

    def gather_surrounding(self):
        '''Data from surrounding spots are collected and formatted into dict data_surrounding. 
        Then all surrounding cell/s data_surrounding dict is merged into own. Extra work is done 
        to obtain details on the cells surrounding own with the same index. Those are labeled friendly.'''
        data = {}
        al = (0, 0, 0, 0, 0) # action_list
        #limits = (0, w.size[0], w.size[1]) # During pruning only positions within grid are passed thorugh.
        surrounding = tuple(self.surround_vector.T)
        other_cells_indices = w.grit['cell'][surrounding]
        other_cells_genes = w.get_genes(surrounding)
        #print(f'other_cells_indices:\n{other_cells_indices}\n')
        for pos, ix, gen in zip(self.surround_vector, other_cells_indices, other_cells_genes):
            corr = self._check_gene_correlation(gen)
            al = self.choose_actions(corr)
            gene_ids = 0
            corr = 0
            data_key = tuple(pos)
            data_val = (pos[0], pos[1], self.index, ix, self.position[0], self.position[1], corr, al[0], al[1], al[2], al[3], al[4])
            data[data_key] = data_val
        self.data_surrounding.update(data)

    @property
    def _express_gene_id(self):
        _gene_blueprint = w.get_new_genes(self)
        #logging.debug(f'Object {self.role}::{self.index} gene id:{_gene_blueprint}.')
        return _gene_blueprint

    def _check_gene_correlation(self, other_gene):
        #n = len(self.gene_id)
        n = 8
        g1 = self.gene_id[:n]
        
        g2 = other_gene[:n]
        if np.any(g2):
            
            #assert len(g1) == len(g2)
            gm1 = np.mean(g1)
            gm2 = np.mean(g2)

            denom = ((np.mean(g1**2) - gm1**2)**0.5) * ((np.mean(g2**2) - gm2**2)**0.5) 
            if denom: # This runs almost twice as fast compared to using np.corrcoef for some reason.
                gene_id_cov = ((g1 - gm1) * (g2 - gm2) ) / denom
            else:
                e = f'_check_gene_correlation:: Object {self.role}::{self.index}, g1:{g1}, g2:{g2}, g1:{gm1}, g2:{gm2}'  
                #logging.error(f'Object {self.role}::{self.index} g1:{g1}, g2:{g2}, g1:{gm1}, g2:{gm2}')
                raise ValueError(e)
            gene_id_cov = np.around(np.sum(gene_id_cov) / n, 3)
        else:
            gene_id_cov = 0
        
        #logging.debug(f'Object {self.role}::{self.index} gene corr:{gene_id_cov}.')
        return gene_id_cov
        #print(f'gene_id_cov: {gene_id_cov}')

    def choose_actions(self, corr):
        '''Input float or list of floats between -1 and 1, output list of BOOLs for actions:
        (share_info, share_food, mate, eat). Action eat is exclusive from any other action.'''
        def commit_action_on_affinity(corr):
            assert corr >= -1 and corr <= 1
            p = (corr + 1) / 2 # Normalising value from -1~1 to 0~1
            p_ = p # Second prob value that will change throughout.
            p_or_n, im_x, fd_x, mt = (0, 0, 0, 0) # Please note the importance of use of underscores.
            try:
                p_or_n = np.random.choice((0, 1), p=((1-p_), p_)) # Friend or foe?
            except ValueError:
                logging.error(f'corr:: {corr}, p:{p_}.')
                raise ValueError
            p_ = p_**2 
            im_x = np.random.choice((0, 1), p=((1-p_), p_)) # Sharing information?
            p_ = p_**2
            fd_x = np.random.choice((0, 1), p=((1-p_), p_)) # Sharing food?
            p_ = p_**2
            mt = np.random.choice((0, 1), p=((1-p_), p_)) # Potential mate?
            p_e = (1-p_or_n) * (1-p)
            eat = np.random.choice((0, 1), p=((1-p_e), p_e)) # Potential dinner?
            #print(f"commit_action_on_affinity:: corr:{corr}, p_e:{p_e}, {(p_or_n, im_x, fd_x, mt, eat)}")
            return (p_or_n, im_x, fd_x, mt, eat)
        if isinstance(corr, (list, tuple)):
            committed = []
            for p_c in corr:
                committed.append(commit_action_on_affinity(p_c))
            return np.array(committed)
        if isinstance(corr, (int, float)):
            committed = commit_action_on_affinity(corr)
            return committed       
        
    def __str__(self):
        s = f'{self.role}::{self.index}::{self.position}::food:'
        s = s + f'{self.food}:: sur dict len:{len(self.data_surrounding)}'   
        s = s + f':: peers:{self.get_peer_data}'              
        return s

w = World(size=(8, 8))
for i in range(63):
    w.spawn(position='random')

TEST = 2
for t in range(TEST):
    w.tick()

for k, v in w.cell_list.items():
    print(v)

fig, ax = plt.subplots()

im = ax.imshow(w.grit['cell'], cmap='hot', interpolation='nearest')
ax.grid(visible=False)
plt.show()      
