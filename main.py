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
        self.over_populated = False
        self.grit = {}
        self.count_of_resources = 4
        self.grit['cell'] = np.zeros(size).astype(int)
        #self.grit['resources'] = np.ones(((self.count_of_resources,) + size)).astype(int)    
        self.grit['cell_alive'] = np.zeros(size).astype(int)
        self.grit['gene_id'] = {}
        self.gene_pool = np.zeros(((self.size_prod,) + self.size)).astype(int)
        self.extinction = False
        
        self.random_spots = np.random.permutation(self.size_prod).reshape(self.size)
        self.set_up_resources()
        logging.debug('--------------------------------------------')
        logging.debug(f'Object {self.role}::Created size:{self.size}.')
        
    def set_up_resources(self, phase1=0.5, phase2=0.5):
        r = np.linspace(0, 1, size[0])
        template_0 = (0.5 - np.abs((r) - phase1)) * 2
        template_1 = 1 - template_0
        template_2 = (((r) - phase2)**2)*4
        template_3 = 1 - (((r) - phase2)**2)*4
        mat = (template_0, template_1, template_2, template_3)
        data = []
        data.append((np.dot(mat[2].reshape(-1, 1), mat[1].reshape(1, -1))*255).astype(int))
        data.append((np.dot(mat[1].reshape(-1, 1), mat[3].reshape(1, -1))*255).astype(int))
        data.append((np.dot(mat[3].reshape(-1, 1), mat[2].reshape(1, -1))*255).astype(int))
        data.append((np.dot(mat[0].reshape(-1, 1), mat[3].reshape(1, -1))*255).astype(int))
        self.grit['resources'] = np.array(data).reshape(-1, self.size[0], self.size[1])

    def tick(self):
        population_copy = self.population_dict.copy()
        for k, v in population_copy.items():
            if v.alive:
                v.tick()
                    
    def spawn(self, parent=None, role='cell', position=None):
        new_index = self._population_counter 
        if new_index > self.size_prod:
            print(f'World is overpopulated')
            self.over_populated = True
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
        #new_genes = np.arange(self.size_prod)
        pos = obj.position
        self.gene_pool[:, pos[0], pos[1]] = new_genes
        return new_genes
        
        
    def get_genes(self, pos):
        #print(f'get_genes:: obj:{obj}, type:{type(obj)}')
        #logging.debug(f'Object {self.role}:: gene pos:{pos}.')
        genes = self.gene_pool[:, pos[0], pos[1]]
        return genes.T

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
        self.gene_pool[:, pos[0], pos[1]] = genes #
        
        
    def activate(self, obj):
        obj_index = obj.index
        if isinstance(obj, Cell):
            self.cell_list[obj_index] = obj
            obj_position = obj.position
            self.grit['cell_alive'][obj_position] = 1
            self.grit['cell'][obj_position] = obj_index
            self.grit['gene_id'][obj.position] = obj.gene_id
            self.set_genes(obj, obj.gene_id)

iff_dict = {0: 'eat', }
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
        self.food_delta = 0
        self.food_share = 0 # Calculated in gauge_wellness
        self.social_vector = np.array([0, 0, 0, 0])
        self.social_delta = np.array([0, 0, 0, 0])
        self.mood_factor = np.array([1,1,1,1])
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

        #logging.debug(f'Object {self.role} created at {position}.')
        w.cell_list[self.index] = self
        w.activate(self)
        
    def tick(self):
        self.gather_surrounding()
        self.gather_food()
        sensory_dict = self.process_sensory
        self.sensory_dict = sensory_dict
        self.gauge_wellness()
        peers_info = sensory_dict['peers_info_exchange']
        peers_food = sensory_dict['peers_food_exchange']
        potential_mates = sensory_dict['peers_mate_potential']
        potential_victims = sensory_dict['peers_eat_potential']
        #logging.debug(f'Object {self.role}::{self.index} sensory_dict: {sensory_dict}.')
        if len(peers_info):
            self.share_info(peers_info)
        if len(peers_food):
            self.share_food(peers_food)
        if len(potential_mates):
            self.mate(potential_mates)
        if len(potential_victims):
            self.eat(potential_victims)
        self.grow()
            
    def gauge_wellness(self):
        new_social_vector = np.array(list((self.get_peer_data).values()))
        self.social_delta = new_social_vector - self.social_vector
        self.social_vector = new_social_vector
        food_new = np.sum(self.food_processed)
        self.food_delta = np.arctan(food_new - self.food)
        #logging.debug(f'Object {self.role}::{self.index} food_delta: {self.food_delta}.')
        self.food = np.around((self.food + food_new), 2)
        self.wellness = (self.social_delta * self.food_delta * self.mood_factor).astype(int)
        #logging.debug(f'Object {self.role}::{self.index} wellness: {self.wellness}.')
        self.food_share = np.around((self.food * self.food_share_factor), 2) #@property
        self.food = np.around((self.food - self.food_share), 2)
        assert self.food >= 0
        #logging.debug(f'Object {self.role}::{self.index} food: {self.food}, food_share: {self.food_share}.')
        
        
    def gather_food(self):
        data = np.array(list(self.data_surrounding.values()), dtype=object)
        data = data[np.where(data[:, 3]==0)]
        
        if len(data):
            pos = data[:, 0:2].T
            #logging.debug(f'Object-1-- {self.role}::{self.index} pos: {pos}, position:{self.position}.')
            l1, l2 = [], []
            l1.append(list(pos[0]) + [self.position[0]])
            l2.append(list(pos[1]) + [self.position[1]])
            pos = (tuple(l1), tuple(l2))
        else:
            pos = self.position
        #logging.debug(f'Object {self.role}::{self.index} pos: {pos}.')
        # Ga hier nu na w.resource ofso en ga da die vreet haal
        mined = w.grit['resources'][:, pos[1], pos[0]] # Needs to be inspected, indexes are reversed!
        logging.debug(f'Object {self.role}::{self.index} mined: {mined}, shape:{np.shape(mined)}.')
        if len(np.shape(mined)) == 3:
            mined = np.sum(mined, axis=2)
        gene_factor = np.random.randint(255, size=4)
        gene_factor = (gene_factor / np.sum(gene_factor)).reshape(-1,1)
        self.food_processed = gene_factor * mined     
        
        
    
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
        empty = data[np.where(data[:, 3]==0)]
        sensory_dict['empty_spots'] = empty
        for i, k in zip(data_col_ix, keys):
            info_ = data[np.where(data[:, i]==1)] # Choose relevant action column
            info = info_[:, 3][np.where(info_[:, 3]>0)] # and discard if spot is empty.
            sensory_dict[k] = info
        #logging.debug(f'Object {self.role}::{self.index} sensory_dict:{sensory_dict}.')
        return sensory_dict
        
        
    def grow(self):        
        # Determine how many to grow
        if w.over_populated:
            return
        genes = np.random.randint(255, size=8)/255
        genes_fact = genes[:4]
        genes_bias = genes[4:]
        grow_count = 0
        #logging.debug(f'Object {self.role}::{self.index} v1:{v}.')
        grow_count = np.sum((self.wellness + genes_bias) * genes_fact)
        grow_count = int(np.max((grow_count, 0.)))
        logging.debug(f'Object {self.role}::{self.index} v3: {grow_count}.')
        #v = np.min((v, 1.))
        
        #w.spawn(parent=self, position=position)
        if grow_count:
            position = self.choose_position(grow_count)
            logging.debug(f'Object {self.role}::{self.index} position:{position}.')
            if not isinstance(position, (int, float)):
                for p in position:
                    #logging.debug(f'Object {self.role}::{self.index} ---p:{p}.')
                    w.spawn(parent=self, position=tuple(p))
    
    def eat(self, peer_list):
        logging.debug(f'Object {self.role}::{self.index}.')
    
    def mate(self, peer_list):
        '''Choose one of mates in peer_list based on its features such as:
        corr, food, age'''
        if w.over_populated:
            return
        #print(f'mate::Object {self.role}::{self.index}:: surr:\n{self.data_surrounding}.\n')
        data = np.array(list(self.data_surrounding.values()), dtype=object )
        mate_data = data[np.where(data[:, 10])]
        #print(f'mate::Object {self.role}::{self.index}:: mate_data:\n{mate_data}.\n')
        
    def choose_position(self, grow_count):
        possible_spots_all = self.sensory_dict['empty_spots']
        #logging.debug(f'Object {self.role}::{self.index}, possible_spots_all:{possible_spots_all}.') 
        possible_spot_count = len(possible_spots_all)
        #logging.debug(f'Object {self.role}::{self.index}, possible_spot_count:{possible_spot_count}.')
        possible_spots = possible_spots_all[:, 0:2]        
        #logging.debug(f'Object {self.role}::{self.index}, possible_spots:{possible_spots}.') 
        #if possible_spot_count == 1:
        chosen_spots = possible_spots
        if possible_spot_count > 1:
            logging.debug(f'Object {self.role}::{self.index}, possible_spot_count:{possible_spot_count}.')
            logging.debug(f'Object {self.role}::{self.index}, grow_count:{grow_count}.')
            
            if possible_spot_count >= grow_count:
                chosen_indices = np.random.choice(possible_spot_count, grow_count, replace=False)
            #logging.debug(f'Object {self.role}::{self.index}, chosen_indices:{chosen_indices}.') 
                chosen_spots = possible_spots[chosen_indices]
                            
        else:
            chosen_spots = 0
        logging.debug(f'Object {self.role}::{self.index}, chosen_spots:{chosen_spots}.')  
        return chosen_spots
        
    def share_info(self, peer_list):
        if isinstance(peer_list, (tuple, list, np.ndarray)) and len(peer_list):
            for c in peer_list:
                w.cell_list[c].data_surrounding.update(self.data_surrounding)
                self.data_surrounding.update(w.cell_list[c].data_surrounding) # Lastly, update ourself
        else:            
            raise TypeError
        
    def share_food(self, peer_list):
        if isinstance(peer_list, (tuple, list, np.ndarray)) and len(peer_list):                   
            collected_food = self.food_share    
            logging.debug(f'Object {self.role}::{self.index} collected_food-1: {collected_food}.')
            for c in peer_list:
                collected_food = collected_food + w.cell_list[int(c)].food_share
            collected_food = collected_food / len(peer_list)   
            logging.debug(f'Object {self.role}::{self.index} collected_food-2: {collected_food}.')
            for c in peer_list:
                w.cell_list[int(c)].food = collected_food
        else:            
            logging.error(f'Object {self.role}::{self.index} peer_list: {peer_list}.')
            raise TypeError
            
            
    @property
    def food_share_factor(self):
        return (np.random.randint(255, size=1)) / 255
                

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
        #logging.debug(f'Object {self.role}::{self.index} other_cells_genes:{other_cells_genes}.')
        #print(f'other_cells_indices:\n{other_cells_indices}\n')
        for pos, ix, gen in zip(self.surround_vector, other_cells_indices, other_cells_genes):
            corr = self._check_gene_correlation(gen)
            corr_byte = int((corr + 1) * 127)
            al = self.choose_actions(corr)
            gene_ids, corr = 0, 0
            data_key = tuple(pos)
            data_val = (pos[0], pos[1], # 0, 1
                        self.index,     # 2
                        ix,             # 3
                        self.position[0], self.position[1], # 4, 5
                        corr_byte, # 6
                        al[0], al[1], al[2], al[3], al[4]) # 7, 8, 9, 10, 11, 12
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
        #logging.debug(f'Object {self.role}::{self.index} g1:{g1}, g2:{g2}.')
        if np.any(g2):            
            #assert len(g1) == len(g2)
            gm1 = np.mean(g1)
            gm2 = np.mean(g2)
            denom = ((np.mean(g1**2) - gm1**2)**0.5) * ((np.mean(g2**2) - gm2**2)**0.5) 
            #logging.debug(f'Object {self.role}::{self.index} denom:{denom}.')
            if denom: # This runs almost twice as fast compared to using np.corrcoef for some reason.
                gene_id_cov = ((g1 - gm1) * (g2 - gm2) ) / denom
                #logging.debug(f'Object {self.role}::{self.index} gene_id_cov:{gene_id_cov}.')
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

for i in range(16):
    w.spawn(position='random')

TEST = 32
for t in range(TEST):
    w.tick()


#for i in range(32):
    #w.spawn(position='random')

    
for k, v in w.cell_list.items():
    print(v)

fig, ax = plt.subplots()

im = ax.imshow(w.grit['cell'], cmap='hot', interpolation='nearest')
ax.grid(visible=False)
plt.show()      
