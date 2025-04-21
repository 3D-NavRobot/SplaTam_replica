import copy

 

from configs.replica.splatam import config as base   # reuse the long one you showed
 


 

config = copy.deepcopy(base)
 


 

# quick scene / resolution tweak
 

config['data']['sequence']              = 'room0'
 

config['data']['desired_image_height']  = 480
 

config['data']['desired_image_width']   = 640
 


 

# tracker
 

config['tracking']['num_iters']         = 15          # will drop to 5 after bootstrap
 

config['mapping']['num_iters']          = 25
 

config['mapping']['use_gaussian_splatting_densification'] = True
 


 

# pruning/densification schedules shorter (mapping_iters==25)
 

config['mapping']['pruning_dict']['stop_after']        = 10
 

config['mapping']['densify_dict']['stop_after']        = 300
 

config['mapping']['densify_dict']['start_after']       = 50
 


 

# mixed precision for RTX‑4000
 

config['mixed_precision'] = True