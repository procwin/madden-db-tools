"""
Roster save tools and utilities
"""

import numpy as np
import pandas as pd

from utils import is_unique, coalesce


def get_ppos_maps():
    '''
    create player postiion dicts
    '''

    ppos_map = {0:'QB', 1:'HB', 2:'FB', 3:'WR', 4:'TE', 5:'LT', 6:'LG', 7:'C', 8:'RG', 9:'RT',
                10:'LE', 11:'RE', 12:'DT', 13:'LOLB', 14:'MLB', 15:'ROLB', 16:'CB', 17:'FS', 18:'SS',
                19:'K', 20:'P', 21:'KR', 22:'PR', 23:'KOS', 24:'LS', 25:'3DRB'}

    ppos_map_r = {v:k for k,v in ppos_map.items()}

    return ppos_map, ppos_map_r


def get_tgid_maps(team):
    '''
    create team id dicts from TEAM table
    '''

    st = team.copy()
    st = st[['tgid','tsna']].drop_duplicates()
    tgid_map = dict(zip(st['tgid'], st['tsna']))
    tgid_map_r = {v:k for k,v in tgid_map.items()}
    
    return tgid_map, tgid_map_r


def find_player(name, play, team, cols=None):
    '''
    player name lookup
    '''

    sp = play.copy()
    st = team.copy()

    if not cols:
        cols = ['pgid','pfna','plna','ppos','pos','pjen','povr','tgid','tsna']

    # extract first and last names
    name = name.lower().strip()
    if " " in name:
        pfna, plna = name.split(" ")
    else:
        pfna = None
        plna = name

    # search play data
    if pfna and plna:
        out = sp.copy().loc[(sp['pfna'].str.lower()==pfna) & (sp['plna'].str.lower()==plna)]
    else:
        out = sp.copy().loc[(sp['plna'].str.lower()==plna)]

    if out.shape[0]>0:
        ppos_map = get_ppos_maps()[0]
        tgid_map = get_tgid_maps(st)[0]
        out['pos'] = [ppos_map[i] for i in out['ppos']]
        out['tsna'] = [tgid_map[i] for i in out['tgid']]
        out = out[cols]
        return out
    else:
        print("No matching players found")


def predict_povr(row, calc_map):
    '''
    get player rating prediction using custom calculation
    '''

    calc = calc_map[row['ppos']]['coef_imp']
    xvars = [i for i in calc_map[row['ppos']]['coef_imp'].index if i != 'Intercept']
    pred_raw = np.dot(row[xvars].values, calc[xvars].values) + calc['Intercept']
    pred_scale = max(0, min(99, pred_raw.round(0)))
    return pred_scale


def predict_pimp(row, calc_map):
    '''
    get player importance prediction using custom calculation
    '''

    pred_raw = calc_map['Intercept'] + \
                        calc_map['povr']*row['povr'] + \
                        calc_map['ddep']*row['ddep'] + \
                        calc_map['ppos'][row['ppos']]
    pred_scale = max(0, min(99, pred_raw.round(0)))
    return pred_scale


def update_dcht(play):
    '''
    update depth chart from PLAY table
    '''

    sp = play.copy()

    # create spec for max depth by position
    ddep_max = {
        0: 3, # qb
        1: 4, # hb
        2: 2, # fb
        3: 6, # wr
        4: 3, # te
        5: 3, # lt
        6: 3, # lg
        7: 3, # c
        8: 3, # rg
        9: 3, # rt
        10: 3, # le
        11: 3, # re
        12: 5, # dt
        13: 3, # lolb
        14: 4, # mlb
        15: 3, # rolb
        16: 5, # cb
        17: 3, # fs
        18: 3, # ss
        19: 2, # k
        20: 2, # p
        21: 1, # kr
        22: 1, # pr
        23: 1, # kos
        24: 1, # ls
        25: 1 # 3drb
    }

    # primary positions, except kicker/punter (0-18) - highest PVOR; break ties with PAWR, PSPD
    cols_dcht = ['tgid','pgid','ppos','pfna','plna','povr','ddep','is_valid']
    cols_dcht_out = ['tgid','pgid','ppos','ddep']

    dcht_sort = sp.copy().loc[(sp['tgid'].isin(range(1,33))) & (sp['ppos'].isin(range(0,19)))]
    dcht_sort.sort_values(by=['tgid','ppos','povr','pawr','pspd'], ascending=[True,True,False,False,False], inplace=True)
    dcht_sort['ddep'] = dcht_sort.groupby(['tgid','ppos']).cumcount()
    dcht_sort['is_valid'] = list(map(lambda x: x[1] <= ddep_max[x[0]]-1, zip(dcht_sort['ppos'], dcht_sort['ddep'])))
    dcht_sort.drop(index=dcht_sort.loc[dcht_sort['is_valid']==False].index, inplace=True)
    dcht_sort = dcht_sort[cols_dcht]

    # k (19) - default to kicker
    dcht_kp = sp.copy().loc[(sp['tgid'].isin(range(1,33))) & (sp['ppos'].isin([19,20]))]
    dcht_k = dcht_kp.sort_values(['tgid','ppos','povr'], ascending=[True,True,False]).groupby(['tgid']).head(1)
    dcht_k['ppos'] = 19
    dcht_k['ddep'] = 0
    dcht_k['is_valid'] = True
    dcht_k = dcht_k[cols_dcht]

    # p (20) - default to punter
    dcht_p = dcht_kp.sort_values(['tgid','ppos','povr'], ascending=[True,False,False]).groupby(['tgid']).head(1)
    dcht_p['ppos'] = 20
    dcht_p['ddep'] = 0
    dcht_p['is_valid'] = True
    dcht_p = dcht_p[cols_dcht]

    # kr/pr (21-22) - highest KRT; break ties with PSPD, PBTK
    dcht_krt = sp.copy().loc[(sp['tgid'].isin(range(1,33)))]
    dcht_krt = dcht_krt.sort_values(['tgid','pkrt','pspd','pbtk'], ascending=[True,False,False,False]).groupby(['tgid']).head(1)
    dcht_krt['ppos'] = 21
    dcht_krt['ddep'] = 0
    dcht_krt['is_valid'] = True
    dcht_krt = dcht_krt[cols_dcht]

    dcht_prt = dcht_krt.copy()
    dcht_prt['ppos'] = 22
    dcht_prt = dcht_prt[cols_dcht]

    # kos (23) - default to kicker
    dcht_kos = dcht_kp.sort_values(['tgid','ppos'], ascending=[True,True]).groupby('tgid').head(1)
    dcht_kos['ppos'] = 23
    dcht_kos['ddep'] = 0
    dcht_kos['is_valid'] = True
    dcht_kos = dcht_kos[cols_dcht]

    # los (24) - lowest POVR tight end
    dcht_los = sp.copy().loc[(sp['tgid'].isin(range(1,33))) & (sp['ppos']==4)]
    dcht_los = dcht_los.sort_values(['tgid','povr'], ascending=[True,True]).groupby('tgid').head(1)
    dcht_los['ppos'] = 24
    dcht_los['ddep'] = 0
    dcht_los['is_valid'] = True
    dcht_los = dcht_los[cols_dcht]

    # 3drb (25) - best pass catching rb
    dcht_trb = sp.copy().loc[(sp['tgid'].isin(range(1,33))) & (sp['ppos']==1)]
    dcht_trb = dcht_trb.sort_values(['tgid','pcth'], ascending=[True,False]).groupby(['tgid']).head(1)
    dcht_trb['ppos'] = 25
    dcht_trb['ddep'] = 0
    dcht_trb['is_valid'] = True
    dcht_trb = dcht_trb[cols_dcht]

    # final depth chart
    out = pd.concat([dcht_sort, dcht_k, dcht_p, dcht_krt, dcht_prt, dcht_kos, dcht_los, dcht_trb], axis=0)
    out.sort_values(['tgid','ppos','ddep'], ascending=[True,True,True], inplace=True)
    out = out[cols_dcht_out]
    out.reset_index(inplace=True, drop=True)

    return out


def get_salary_ref(play):
    '''
    create (yearly) salary reference tables by position/rating
    '''

    sp = play.copy()

    # filter out free agents (zero salary) and create yearly salary fields
    sp_sal = sp.copy().loc[sp['tgid'].isin(range(1,33))]
    sp_sal['ptsa_yr'] = sp_sal['ptsa']/sp_sal['pcon']
    sp_sal['psbo_yr'] = sp_sal['psbo']/sp_sal['pcon']
    sp_sal['povr_grp'] = np.floor(sp_sal['povr']/10)

    # get salary mean/median by ppos and povr rating decile
    sal_ = sp_sal.groupby(['ppos','povr_grp'])
    sal = sal_.size().reset_index().rename(columns={0:'cnt'})
    sal['ptsa_med'] = sal_['ptsa_yr'].median().values
    sal['ptsa_mu'] = sal_['ptsa_yr'].mean().values
    sal['psbo_med'] = sal_['psbo_yr'].median().values
    sal['psbo_mu'] = sal_['psbo_yr'].mean().values

    # impute missing ratings deciles
    ppos_ = pd.DataFrame(range(0,21), columns=['ppos'])
    povr_ = pd.DataFrame(np.array(range(0,10)), columns=['povr_grp'])
    salref_ = ppos_.merge(povr_, how='cross')
    salref = salref_.merge(sal, on=['ppos','povr_grp'], how='left')

    # adjust final reference salary and signing bonus values
    salref['ptsa_adj'] = np.ceil(salref['ptsa_med'])
    salref.loc[salref['povr_grp']==5, 'ptsa_adj'] = 30
    salref['psbo_adj'] = np.ceil(salref['psbo_med'])
    salref.loc[salref['povr_grp']==5, 'psbo_adj'] = 1
    salref.loc[salref['povr_grp']==6, 'psbo_adj'] = 4
    salref.loc[salref['povr_grp']==7, 'psbo_adj'] = 7
    salref.loc[salref['povr_grp']<5, 'ptsa_adj'] = 20
    salref.loc[salref['povr_grp']<5, 'psbo_adj'] = 0

    # create ref table for minimum salary by years of service
    yp_ = range(0,26)
    ms_ = np.append([20,30,40,45,55,55,55,65,65,65,75], np.repeat(75,15))
    salmin = pd.DataFrame({'pyrp':yp_,'ptsa_min': ms_})

    return salref, salmin


def update_salary(row, years, sal_ref, sal_min):
    '''
    update player salary using reference tables
    '''

    # extract data for salary table
    povrg = np.floor(row['povr']/10)
    ppos = row['ppos']
    pyrp = row['pyrp']
    pcon, pvco, pcyl = years, years, years

    # get salary and bonus reference values
    ptsa_, psbo_ = sal_ref.loc[(sal_ref['ppos']==ppos) & (sal_ref['povr_grp']==povrg), ['ptsa_adj','psbo_adj']].values[0]
    minsal_ = sal_min.loc[sal_min['pyrp']==pyrp, 'ptsa_min'].values[0]

    # get final salary and bonus
    ptsa = years*max(ptsa_, minsal_)
    psbo = years*psbo_
    pvts = ptsa
    pvsb = psbo
    pywt = 0

    # update values and return data row
    cols = ['ptsa','pvts','psbo','pvsb','pcon','pvco','pcyl']
    for col in cols:
        row[col] = eval(col)

    return row


def resolve_jersey_dups(play):
    '''
    find and resolve teammates with same jersey number
    '''

    sp = play.copy()
    sptm = sp.copy().loc[sp['tgid'].isin(range(1,33))]

    # get jersey numbers in use by team
    pjen_map_ = sptm.groupby(['tgid'])['pjen'].unique().reset_index()
    pjen_map = {i[0]:sorted(int(j) for j in i[1]) for i in pjen_map_.values}

    # get dups
    dups_ = is_unique(sptm, ['tgid','pjen'], print_dups=False, return_dups=True)

    # exclude best player by tgid/pjen (i.e. give them the number)
    cols = ['tgid','pgid','pfna','plna','pjen','ppos','povr']
    dups = sptm.merge(dups_, on=['tgid','pjen'], how='inner')[cols]
    dups.sort_values(['tgid','pjen','povr'], ascending=[1,1,1], inplace=True)
    keep = dups.groupby(['tgid','pjen'], sort=False).tail(1).index
    dups = dups.drop(index=keep)

    # assign new number
    for idx in dups.index:
        curr = dups.loc[idx]
        # attempt to assign within current decile; fallback to any valid number
        pjen_low = int(np.floor(curr['pjen']/10))
        pjen_range = range(pjen_low*10, pjen_low*10+10)
        in_use = pjen_map[curr['tgid']]
        try:
            pjen_new = next(i for i in pjen_range if i not in in_use)
        except StopIteration:
            pjen_new = next(i for i in range(1,99) if i not in in_use)
        # add new number to in-use map and update dups data
        pjen_map[curr['tgid']].append(pjen_new)
        dups.loc[idx, 'pjen'] = pjen_new

    # update play data
    upd = sp.merge(dups[['pgid','pjen']], on='pgid', how='left', suffixes=[None, '_upd'])
    upd['pjen'] = coalesce(upd, 'pjen_upd', 'pjen', impute=np.nan)
    upd = upd[sp.columns]

    return upd


def validate_play_table(play, team, ddplay, rate_calc):
    '''
    validate play data
    '''

    sp = play.copy()
    st = team.copy()
    dp = ddplay.copy()
    ppos_map = get_ppos_maps()[0]
    tgid_map = get_tgid_maps(st)[0]

    sp_tm = sp.loc[sp['tgid'].isin(range(0,33))]
    sp_fa = sp.loc[sp['tgid'] == 1009]
    rc = rate_calc.copy()

    # ensure unique player ids
    is_unique(sp, ['pgid'], print_dups=True)
    is_unique(sp, ['poid'], print_dups=True)

    # ensure unique name/position
    is_unique(sp, ['pfna','plna','ppos'], print_dups=True)

    # ensure unique jersey number by team
    is_unique(sp_tm, ['tgid','pjen'], print_dups=True)

    # ensure valid positions
    bad_ppos = sp.loc[~sp['ppos'].isin(range(0,21)) | (~sp['ppos']==sp['pops']), ['pgid','pfna','plna','ppos','pops']]
    if bad_ppos.shape[0] > 0:
        print(f"Invalid POS/PPOS:\n{bad_ppos}\n")

    # check for missing values
    missing = pd.isnull(sp).sum(axis=0)
    has_miss = list(missing.loc[missing>0].index)

    if len(has_miss)>0:
        print(f"Columns with missing values:\n{', '.join([i for i in has_miss])}\n")

    # ensure valid ranges
    cols_num = sp.select_dtypes([np.number]).columns.values
    ranges_ = dp.loc[dp['column'].str.lower().isin(cols_num), ['column','range_obs']].values
    ranges = {row[0].lower(): range(eval(row[1])[0], eval(row[1])[1]+1) for row in ranges_}

    bad_range = []
    for col in cols_num:
        if sp.loc[~sp.phgt.isin(ranges['phgt'])].shape[0]>0:
            bad_range.append(col)

    if len(bad_range)>0:
        print(f"Columns with invalid range:\n{', '.join([i for i in bad_range])}\n")

    # check obs v. expected ratings
    diff_thresh = 3
    cols_rate = ['pgid','pfna','plna','ppos','tgid','povr','povr_pred','povr_diff']
    sp['povr_pred'] = sp.apply(predict_povr, calc_map=rc, axis=1)
    sp['povr_diff'] = sp[['povr','povr_pred']].diff(axis=1).iloc[:,1]
    rate_thresh = sp.copy().loc[sp.povr_diff.abs()>=diff_thresh, cols_rate]
    if rate_thresh.shape[0]>0:
        print(f"Players with absolute value of rating difference (obs-pred) >= {diff_thresh}\n{rate_thresh}:\n")

    # ensure zero salary for free agents
    cols_salary = ['ptsa','pvts','psbo','pvsb','pcon','pvco','pcyl']
    idx_bad_sal = []
    for col in cols_salary:
        idx_bad_sal += list(sp_fa.loc[sp_fa[col]>0].index)
    idx_bad_sal = list(set(idx_bad_sal))
    if len(idx_bad_sal) > 0:
        bad_sal_fa = sp_fa.loc[idx_bad_sal, ['pgid','pfna','plna','tgid']+cols_salary]
        print(f"Free Agents with nonzero salary:\n{bad_sal_fa}\n")

    # ensure positive salary for rostered players
    cols_salary = ['ptsa','pvts','pcon','pvco']
    idx_bad_sal = []
    for col in cols_salary:
        idx_bad_sal += list(sp_tm.loc[sp_tm[col]==0].index)
    idx_bad_sal = list(set(idx_bad_sal))
    if len(idx_bad_sal) > 0:
        bad_sal = sp_tm.loc[idx_bad_sal, ['pgid','pfna','plna','tgid']+cols_salary]
        print(f"Rostered players with zero salary:\n{bad_sal}\n")

    # ensure roster depth
    ppos_ = pd.DataFrame(range(0,21), columns=['ppos'])
    tgid_ = pd.DataFrame(range(1,33), columns=['tgid'])
    base_ = tgid_.merge(ppos_, how='cross')

    ppos_cnt_ = sp.groupby(['tgid','ppos']).size().reset_index().rename(columns={0:'cnt'})
    ppos_cnt = base_.merge(ppos_cnt_, on=['tgid','ppos'], how='left')
    ppos_cnt['tsna'] = [tgid_map[i] for i in ppos_cnt['tgid']]
    ppos_cnt['pos'] = [ppos_map[i] for i in ppos_cnt['ppos']]
    ppos_cnt.fillna(0, inplace=True)
    ppos_cnt = ppos_cnt[['tgid','tsna','ppos','pos','cnt']]

    ppos_cnt_thresh = ppos_cnt.copy().loc[ppos_cnt['cnt']<1]
    if ppos_cnt_thresh.shape[0]>0:
        print(f"Teams with no players at position:\n{ppos_cnt_thresh}\n")

    # ensure valid roster size
    roster_size = sp_tm.groupby('tgid').size().reset_index().rename(columns={0:'cnt'})
    roster_size['abs_diff'] = [abs(53-c) for c in roster_size['cnt']]
    roster_size['tsna'] = [tgid_map[i] for i in roster_size['tgid']]
    
    roster_thresh = 3
    roster_size_thresh = roster_size.loc[roster_size['abs_diff']>roster_thresh, ['tgid','tsna','cnt']]
    if roster_size_thresh.shape[0]>0:
        print(f"Teams under/over roster limit (53) by {roster_thresh+1}+ players:\n{roster_size_thresh}\n")
