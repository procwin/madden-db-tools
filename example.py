"""
Roster update workflow
"""

import numpy as np
import pandas as pd
import yaml
from save_updater import Save


'''
instantiate save from config file
'''

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

save = Save(config)


'''
quick tools

- reset(): restore imported save to load state
- search_player(): find player by name; single name will be treated as last name
'''

#save.reset()
save.search_player('ray lewis')
save.search_player('lewis')


'''
run base updates

- can be run individually (commented out) or in batch via run_base_updates()
'''

#save._update_missing_bios(write=True)
#save._add_caps(write=True)
#save._drop_players(write=True)
#save._remove_injuries(write=False)
save.run_base_updates(write=True)


'''
process transactions

- run_tx_execute(): perform roster transactions
'''

# execute tx
save.run_tx_execute(write=True)


'''
apply player/team updates

- update_ratings_custom(): update player ratings based on external data
- update_salaries(): update salaries by transaction type
- reorder_dcht(): reorder depth charts by highest overall rating
'''

# apply custom ratings updates
save.update_ratings_custom(write=True)

# update salaries
save.update_salaries(write=True)

# reorder depth charts
save.reorder_dcht(write=True)


'''
apply final updates

- update_pimp(): update player importance
- resolve_jersey_duplicates(): resolve jersey number clashes
- validate_play(): run player table data validation checks (informational)
- export_tables(): export save tables
'''

# update player importance
save.update_pimp(write=True)

# resolve jersey duplicates
save.resolve_jersey_duplicates(write=True)

# validate
save.validate_play()

# export
save.export_tables()
