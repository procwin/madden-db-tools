"""
Roster save management class
"""

import os
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from utils import coalesce, to_numeric, format_data

from save_tools import get_ppos_maps, get_tgid_maps, find_player, predict_povr, predict_pimp, \
                        update_dcht, get_salary_ref, update_salary, resolve_jersey_dups, validate_play_table


class Save():
    '''
    Roster save data management tool
    '''

    global coalesce, to_numeric, format_data
    global get_ppos_maps, get_tgid_maps, find_player, predict_povr, predict_pimp, \
            update_dcht, get_salary_ref, update_salary, resolve_jersey_dups, validate_play_table


    def __init__(self, config):
        
        self.config = config
        self._init_data()
        self._init_tools()


    def __str__(self):

        return 'Save object'


    def _init_data(self):
        '''
        load data from config
        '''

        '''saves and data dicts'''
        # set load paths from config
        config = self.config
        dd_path = f"{config['setup']['dir']}/{config['setup']['data_dict']}"
        save_dir = config['saves']['dir']
        save_name = config['saves']['import']

        # load and process data
        saves = {
            'play': {'sv': None, 'dd': None, 'cols_sort': ['tgid','ppos','pgid']},
            'team': {'sv': None, 'dd': None, 'cols_sort': ['tgid']},
            'dcht': {'sv': None, 'dd': None, 'cols_sort': ['tgid','ppos','ddep']},
            'injy': {'sv': None, 'dd': None, 'cols_sort': ['tgid','pgid']}
        }

        for key in saves.keys():
            dd = pd.read_excel(dd_path, sheet_name=key.upper())
            sv = pd.read_csv(f"{save_dir}/{save_name}/{save_name}_{key.upper()}.csv")
            cols_out = dd.sort_values(by='view_id', ascending=True)['column'].str.lower().values
            cols_sort = saves[key]['cols_sort']
            sv = format_data(sv)[cols_out]
            sv.sort_values(cols_sort, inplace=True)
            sv.reset_index(inplace=True, drop=True)
            saves[key]['dd'] = dd
            saves[key]['sv'] = sv

        # save data to instance objects
        self.sv_play = saves['play']['sv']
        self.dd_play = saves['play']['dd']
        self.sv_team = saves['team']['sv']
        self.dd_team = saves['team']['dd']
        self.sv_dcht = saves['dcht']['sv']
        self.dd_dcht = saves['dcht']['dd']
        self.sv_injy = saves['injy']['sv']
        self.dd_injy = saves['injy']['dd']

        '''updates'''
        upd_dir = config['updates']['dir']

        # missing bios
        miss_path_ = config['updates']['miss']
        if miss_path_:
            miss_path = f"{upd_dir}/{miss_path_}"
            self.upd_miss = format_data(pd.read_csv(miss_path))

        # cap additions
        caps_path_ = config['updates']['caps']
        if caps_path_:
            caps_path = f"{upd_dir}/{caps_path_}"
            self.upd_caps = format_data(pd.read_csv(caps_path))

        # player deletions
        drop_path_ = config['updates']['drop']
        if drop_path_:
            drop_path = f"{upd_dir}/{drop_path_}"
            self.upd_drop = format_data(pd.read_csv(drop_path))

        # transactions
        txss_path_ = config['updates']['txss']
        if txss_path_:
            txss_path = f"{upd_dir}/{txss_path_}"
            upd_tx = format_data(pd.read_csv(txss_path))
            upd_tx['date'] = pd.to_datetime(upd_tx['date'], format='%Y-%m-%d')
            self.upd_tx = upd_tx.copy()

        # ratings updates
        rate_path_ = config['updates']['rate']
        if rate_path_:
            rate_path = f"{upd_dir}/{rate_path_}"
            upd_rate = format_data(pd.read_csv(rate_path))
            self.upd_rate = upd_rate.copy()


    def _init_tools(self):
        '''
        load save tools
        '''

        config = self.config
        
        # position map
        self.ppos_maps = get_ppos_maps()

        # team map
        self.tgid_maps = get_tgid_maps(self.sv_team)

        # povr ratings calculator
        povr_calc_d = config['setup']['dir']
        povr_calc_f = config['setup']['povr_calc']
        with open(f"{povr_calc_d}/{povr_calc_f}", 'rb') as file:
            self.povr_calc = pickle.load(file)

        # pimp ratings calculator
        pimp_calc_d = config['setup']['dir']
        pimp_calc_f = config['setup']['pimp_calc']
        with open(f"{pimp_calc_d}/{pimp_calc_f}", 'rb') as file:
            self.pimp_calc = pickle.load(file)


    def search_player(self, name, cols=None):
        '''
        player name search (wrapper)
        '''

        return find_player(name, self.sv_play, self.sv_team, cols=cols)


    def _update_missing_bios(self, play=None, write=False):
        '''
        update missing bios for default players
        '''

        # load data
        if play is None:
            sp = self.sv_play.copy()
        else:
            sp = play.copy()
        st = self.sv_team.copy()
        su = self.upd_miss.copy()
        team_map = get_tgid_maps(st)[1]

        # join play and update data
        su['tgid'] = list(map(lambda x: team_map[x], su['tsna']))
        out = sp.merge(su, on=['tgid','pfna','plna'], how='left')
        cols_upd = ['pfna_upd','plna_upd'] #,'pjen_upd']
        col_pairs = [[cu.split('_')[0], cu] for cu in cols_upd]

        # coalesce columns
        for cp in col_pairs:
            out[cp[0]] = coalesce(out, cp[1], cp[0], impute=np.nan)
        out = out[sp.columns]

        if write:
            self.sv_play = out.copy()
        else:
            return out
        

    def _add_caps(self, play, write=False):
        '''
        add CAPS to play data
        '''

        if play is None:
            sp = self.sv_play.copy()
        else:
            sp = play.copy()
        uc = self.upd_caps.copy()

        # ensure same data frame columns
        cols_same = all(sp.columns == uc.columns)
        if not cols_same:
            raise Exception("Data frames do not have the same columns")
        
        # merge and update
        out = pd.concat([sp, uc], axis=0)
        out.sort_values(['tgid','ppos','pgid'], inplace=True)
        out.reset_index(inplace=True, drop=True)

        if write:
            self.sv_play = out.copy()
        else:
            return out


    def _drop_players(self, play, write=True):
        '''
        delete players from game
        '''

        if play is None:
            sp = self.sv_play.copy()
        else:
            sp = play.copy()
        drop = self.upd_drop.copy()
        drop['delete'] = 1
        out = sp.merge(drop,
                       left_on=['pgid', sp['pfna'].str.lower(), sp['plna'].str.lower().str.strip()],
                       right_on=['pgid', drop['pfna'].str.lower(), drop['plna'].str.lower().str.strip()],
                       how='left')
        idx_drop = out.loc[out['delete']==1].index
        sp.drop(index=idx_drop, inplace=True)

        if write:
            self.sv_play = sp.copy()
        else:
            return sp
        

    def _remove_injuries(self, write=False):
        '''
        remove preexisting injuries, if any
        '''

        si = self.sv_injy.copy()
        si.drop(index=si.index, inplace=True)

        if write:
            self.sv_injy = si.copy()
        else:
            return si


    def run_base_updates(self, write=False):
        '''
        apply inital updates for PLAY, INJY tables
        '''

        sp = self.sv_play.copy()
        si = self.sv_injy.copy()

        # PLAY updates
        # update missing bios
        bio = self._update_missing_bios(sp, write=False)
        # add caps
        caps = self._add_caps(bio, write=False)
        # delete players from game
        out = self._drop_players(caps, write=False)

        # INJY updates
        # remove injuries
        si = self._remove_injuries(write=False)

        if write:
            self.sv_play = out.copy()
            self.sv_injy = si.copy()
        else:
            return out


    def update_ratings_custom(self, write=False):
        '''
        apply custom player ratings updates from external data
        '''

        sp = self.sv_play.copy()
        ur = self.upd_rate.copy()
        dp = self.dd_play.copy()

        # merge new ratings
        rate = sp.merge(ur, on='pgid', how='left', suffixes=[None, '_upd'])

        cols_attr = dp.loc[dp['category'].str.lower()=='attributes', 'column'].str.lower().values
        cols_sp = dp['column'].str.lower().values

        # coalesce ratings
        for col in cols_attr:
            rate[col] = coalesce(rate, col+'_upd', col, impute=np.nan)

        rate = rate[cols_sp]

        # update overall
        rate['povr'] = rate.apply(predict_povr, calc_map=self.povr_calc, axis=1)

        if write:
            self.sv_play = rate.copy()
        else:
            return rate


    def run_tx_execute(self, write=False):
        '''
        execute finalized tx on play data
        '''
        
        sp = self.sv_play.copy()
        txfn = self.upd_tx.copy()

        # merge play and tx data
        cols_tx = ['pgid','tx','tgid_fr','tgid_to']
        sp_tx = sp.merge(txfn[cols_tx], on='pgid', how='left')

        # update team id
        sp_tx['tgid'] = coalesce(sp_tx, 'tgid_to','tgid', impute=np.nan)
        sp_tx['ppti'] = coalesce(sp_tx, 'tgid_fr','tgid', impute=np.nan)

        # update years with team
        to_fa = ['release','waive','practice','retire']
        to_tgid = ['sign','resign','trade']
        sp_tx.loc[sp_tx['tx'].isin(to_fa), 'pywt'] = 31
        sp_tx.loc[sp_tx['tx'].isin(to_tgid + ['trade']), 'pywt'] = 0

        # output
        sp_tx.drop(columns=cols_tx[1:], inplace=True)

        if write:
            self.sv_play = sp_tx.copy()
        else:
            return sp_tx


    def update_salaries(self, write=False):
        '''
        update salaries (wrapper)
        '''

        sp = self.sv_play.copy()
        cols_salary = ['ptsa','pvts','psbo','pvsb','pcon','pvco','pcyl']        

        # zero out free agent contracts
        sp.loc[sp['tgid']==1009, cols_salary] = 0

        # generate salary for rostered players without contracts
        salref, salmin = get_salary_ref(sp)
        idx_nosal = sp.loc[(sp['tgid'].isin(range(1,33))) & (sp['ptsa']==0)].index
        sp.loc[idx_nosal, :] = sp.loc[idx_nosal].apply(update_salary, years=3, sal_ref=salref, sal_min=salmin, axis=1)

        if write:
            self.sv_play = sp.copy()
        else:
            return sp


    def reorder_dcht(self, write=False):
        '''
        reorder depth charts (wrapper)
        '''

        sp = self.sv_play.copy()
        sd = update_dcht(sp)

        if write:
            self.sv_dcht = sd.copy()
        else:
            return sd


    def update_pimp(self, write=False):
        '''
        update player importance (wrapper)
        '''

        sp = self.sv_play.copy()
        sd = self.sv_dcht.copy()

        imp = sp.merge(sd, on=['pgid','ppos','tgid'], how='left')
        imp.loc[pd.isnull(imp['ddep']), 'ddep'] = 0
        imp['pimp'] = imp.apply(predict_pimp, calc_map=self.pimp_calc, axis=1)
        imp.drop(columns=['ddep'], inplace=True)

        if write:
            self.sv_play = imp.copy()
        else:
            return imp


    def resolve_jersey_duplicates(self, write=False):
        '''
        handle duplicate jersey numbers (wrapper)
        '''

        sp = self.sv_play.copy()
        upd = resolve_jersey_dups(sp)

        if write:
            self.sv_play = upd.copy()
        else:
            return upd


    def validate_play(self, play=None, team=None, ddplay=None):
        '''
        validate play data (wrapper)
        '''

        sp = self.sv_play.copy() if play is None else play.copy()
        st = self.sv_team.copy() if team is None else team.copy()
        dp = self.dd_play.copy() if ddplay is None else ddplay.copy()
        rc = self.povr_calc.copy()
        validate_play_table(sp, st, dp, rc)
        

    def export_tables(self):
        '''
        format tables and export
        '''

        # create save folder
        path_saves = self.config['saves']['dir']
        path_export_ = self.config['saves']['export']

        if not path_export_:
            raise Exception("Missing save name: config->saves->export")

        path_export = f"{path_saves}/{path_export_}"
        if not os.path.exists(path_export):
            os.mkdir(path_export)

        # restore original columns and data types
        cols_play = list(self.dd_play['column'].values)
        cols_team = list(self.dd_team['column'].values)
        cols_dcht = list(self.dd_dcht['column'].values)
        cols_injy = list(self.dd_injy['column'].values)

        sp = to_numeric(self.sv_play.copy(), dtype='integer')[[col.lower() for col in cols_play]]
        sp.sort_values(['tgid','ppos','pgid'], inplace=True)
        sp.columns = cols_play

        st = to_numeric(self.sv_team.copy(), dtype='integer')[[col.lower() for col in cols_team]]
        st.sort_values(['tgid'], inplace=True)
        st.columns = cols_team

        sd = to_numeric(self.sv_dcht.copy(), dtype='integer')[[col.lower() for col in cols_dcht]]
        sd.sort_values(['tgid','ppos','ddep'], inplace=True)
        sd.columns = cols_dcht

        si = to_numeric(self.sv_injy.copy(), dtype='integer')[[col.lower() for col in cols_injy]]
        si.sort_values(['tgid','pgid'], inplace=True)
        si.columns = cols_injy

        # write to csv
        sp.to_csv(f"{path_export}/{path_export_}_PLAY.csv", index=0, header=True)
        st.to_csv(f"{path_export}/{path_export_}_TEAM.csv", index=0, header=True)
        sd.to_csv(f"{path_export}/{path_export_}_DCHT.csv", index=0, header=True)
        si.to_csv(f"{path_export}/{path_export_}_INJY.csv", index=0, header=True)

        print(f"Exported save data to: {path_export}")


    def reset(self):
        '''
        resent instance to init
        '''

        self._init_data()
        self._init_tools()
