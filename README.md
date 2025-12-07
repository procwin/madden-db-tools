<div align="center">
    <h1>madden-db-tools-light</h1>
</div>

<p align="center">
    <strong>Python tools for updating Madden NFL 2005 (PS2) rosters</strong>
</p>

---

### Overview
- Modular functionality built around core roster save database tables (PLAY, TEAM, DCHT, INJY)
- Fully automated integrations (e.g. updating depth charts, ratings calculators)
- Partial automations via external data (e.g. custom ratings, Create-A-Player (CAP) additions)
- Roster save-compatible exports for updated tables

**Note**: Scope is Madden NFL 2005 (PS2) and 2004-05 season rosters; code was not tested on other Madden versions/platforms

### Requirements
- Python >=3.8 + dependencies in *requirements.txt*
- External update data (see */updates* folder)

### Quickstart
1. Clone repo and install Python + dependencies
2. Run *example.py* interactively or in terminal: `python example.py`
3. See updated roster tables in */saves/UPDATED*
4. Use [MXDBE](https://www.footballidiot.com/forum/viewtopic.php?t=21400) or equivalent tool to create a roster save using updated tables

### Contents
- */setup*: Data dictionaries and ratings calculators
- */saves:* Import/export roster save data
    - Use save label as subdirectory name and table prefix (e.g. */SAVE* -> */SAVE/SAVE_{PLAY,TEAM,INJY,DCHT}.csv*)
    - Default roster data available in */saves/DEFAULT*
- */updates:* External data used to update roster tables
    - Follow file templates to add/edit/delete records as desired
    - Included files assume baseline **default** roster:
        - *PLAY_MISS_UPD.csv*: Name updates; name information for default unnamed players
        - *PLAY_CAPS_UPD.csv*: (Sample) CAP additions; see PLAY data dictionary for detailed column information
        - *PLAY_DROP_UPD.csv*: (Sample) Player removal; add player info to remove them from game
        - *PLAY_RATE_UPD.csv*: (Sample) Ratings updates; add player info and new attribute values (POVR is auto recalculated)
        - *PLAY_TXS_UPD*.csv*: Preseason transactions; each file includes transactions to execute through file suffix date
- *config.yaml:* Folders, tools, and update data used during execution
    - Defines artifacts to load at runtime from */setup*, */saves*, and */updates*
    - */updates* files are optional; all others are required
    - Default file is parameterized with included tools and data
- *utils.py*: Python tools and utilities
    - Basic data frame operations
- *save_tools.py*: Roster tools and utilities
    - Core roster update logic and functionality
- *save_updater.py*: Main roster update class
    - Instantiates with *config.yaml*
    - Class methods perform specific roster updates
    - Exports updated roster tables to */saves*
- *example.py*: Example execution
    - Demonstrates workflow of core update tools
