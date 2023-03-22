__author__ = "Seolhee Cho"

# GDP RGEP (Reliability-constrainted Generation Expansion Planning) model
# Generation expansion planning model with reliability evaluation
# Objective function is to minimize capital cost, operating cost, and reliability-related penalties
# Nested disjunction --> Disjunct form, others disjunctions --> algebraic form

import pyomo.environ as pyo
from pyomo.gdp import Disjunct, Disjunction
from input_data import dataset

def RGEP_model(transformation, data):
    m = pyo.ConcreteModel()
    
    # Sets    
    m.region = pyo.Set(initialize=data['region']) # index r, set of regions
    m.plant = pyo.Set(initialize=data['plant'])   # index k, set of plants
    m.new_plant = pyo.Set(initialize=data['new_plant'])  # set of potential plants 
    m.old_plant = pyo.Set(initialize=data['old_plant'])  # set of existing plants 
    m.res_plant = pyo.Set(initialize=data['res_plant'])  # set of renewable plants
    m.dpt_plant = pyo.Set(initialize=data['dpt_plant'])  # set of dispathable plants     
    m.rdn_plant = pyo.Set(initialize=data['rdn_plant'])  # set of plants with redundancy
    m.nrd_plant = pyo.Set(initialize=data['nrd_plant'])  # set of plants without redundancy
    m.rdn_old_plant = pyo.Set(initialize=data['rdn_old_plant']) # set of existing plants with redundancy
    m.rdn_new_plant = pyo.Set(initialize=data['rdn_new_plant']) # set of potential plants with redundancy
    m.nrd_old_plant = pyo.Set(initialize=data['nrd_old_plant']) # set of existing plants without redundancy
    m.nrd_new_plant = pyo.Set(initialize=data['nrd_new_plant']) # set of potential plants without redundancy
      
    m.rg_plt = pyo.Set(initialize=data['rg_plt']) # set of plants in regions
    m.rg_plt_pn = pyo.Set(initialize=data['rg_plt_pn']) # set of potential power plants in regions   
    m.rg_plt_ex = pyo.Set(initialize=data['rg_plt_ex']) # set of existing power plants in regions
    m.rg_plt_ex_re = pyo.Set(initialize=data['rg_plt_ex_re']) # set of existing renewable plants in regions
    m.rg_plt_ex_cv = pyo.Set(initialize=data['rg_plt_ex_cv']) # set of existing conventional plants in regions'
    m.rg_plt_rn = pyo.Set(initialize=data['rg_plt_rn']) # set of plants with redundancy in regions
    m.rg_plt_nd = pyo.Set(initialize=data['rg_plt_nd']) # set of plants without redundancy in regions    
    m.rg_plt_rw = pyo.Set(initialize=data['rg_plt_rw']) # set of renewable plants in regions
    m.rg_plt_dp = pyo.Set(initialize=data['rg_plt_dp']) # set of dispatchable plants in regions
    m.rg_plt_dp_pn = pyo.Set(initialize=data['rg_plt_dp_pn']) # set of dispatchable potential power plants in regions  
    m.rg_plt_dp_ex = pyo.Set(initialize=data['rg_plt_dp_ex']) # set of dispatchable existing power plants in regions
      
    m.prgen = pyo.Set(initialize=data['prgen'])  # Parallel generator, index j
    m.strge = pyo.Set(initialize=data['strge'])  # Parallel bettery, index i 
    m.size = pyo.Set(initialize=data['size'])   # Discrete size, index c, 1: small, 2: medium, 3: large
    m.state = pyo.Set(initialize=data['state'])  # Failute states, index s 
    m.year = pyo.Set(initialize=data['year'])   # Planning period, index t
    m.rpdn = pyo.Set(initialize=data['rpdn'])   # Representative days, index n
    m.sub = pyo.Set(initialize=data['sub'])    # Subperiod (hours), index b


    # Scalars
    m.alpha = pyo.Param(initialize=data['alpha'], doc = 'Unmet demand penalty')  
    m.beta = pyo.Param(initialize=data['beta'], doc = 'Downtime penalty')    
    m.OT  = pyo.Param(m.rpdn, m.sub, initialize = data['OT'], doc = 'Operation time (hours)')    
    m.CPF = pyo.Param(m.res_plant, m.year, m.rpdn, m.sub, initialize = data['cpf'], doc = 'Capacity factor (%)')
    m.D = pyo.Param(m.region, m.year, m.rpdn, m.sub, initialize = data['da'], doc = 'Power demand (MWh/subperiod)')    
    m.PRE = pyo.Param(m.old_plant, initialize = data['pre'], doc = 'Preinstalled capacity (MW)')  
    m.URP = pyo.Param(m.plant, initialize = data['urp'], doc = 'Unit reliability (1/failure rate)')
    m.URG  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['urg'], doc = 'Unit reliability of parallel gens. (1/failure rate)')      
    m.CPP = pyo.Param(m.new_plant, m.size, initialize = data['cpp'], doc = 'Nameplate capacity of potential plants (MW)') 
    m.CPG = pyo.Param(m.rdn_plant, m.prgen, m.size, initialize = data['cpg'], doc = 'Nameplate capacity of parallel gens. (MW)')
    m.CPS = pyo.Param(initialize=data['CPS'], doc = 'Nameplate capacity of storage (MW)')     
    m.MINP = pyo.Param(m.dpt_plant, initialize = data['minp'], doc = 'Minimum operation ratio of plants (%)') 
    m.MING = pyo.Param(m.rdn_plant, m.prgen, initialize = data['ming'], doc = 'Minimum operation ratio of parallel gens. (%)') 
    m.MINS = pyo.Param(m.res_plant, m.strge, initialize = data['mins'], doc = 'Minimum operation ratio of batteries (%)')
    m.MAXP = pyo.Param(m.dpt_plant, initialize = data['maxp'], doc = 'Maximum operation ratio of plants (%)') 
    m.MAXG = pyo.Param(m.rdn_plant, m.prgen, initialize = data['maxg'], doc = 'Maximum operation ratio of parallel gens. (%)') 
    m.MAXS = pyo.Param(m.res_plant, m.strge, initialize = data['maxs'], doc = 'Maximum operation ratio of batteries (%)')
    m.ls = pyo.Param(initialize=data['ls'], doc = 'Electricity loss in battery')   
    m.CHE = pyo.Param(initialize=data['CHE'], doc = 'Charging efficiency')      
    m.DCE = pyo.Param(initialize=data['DCE'], doc = 'Discharging efficiency')     
    m.DMIN = pyo.Param(initialize=data['DMIN'], doc = 'Minimum discharging ratio')   
    m.DMAX = pyo.Param(initialize=data['DMAX'], doc = 'Maximum discharging ratio') 
    m.CMIN = pyo.Param(initialize=data['CMIN'], doc = 'Minimum charging ratio')    
    m.CMAX = pyo.Param(initialize=data['CMAX'], doc = 'Maximum charging ratio')   
             
    m.CCP = pyo.Param(m.new_plant, m.size, initialize = data['ccp'], doc = 'Capital cost of potential plants (M$/unit)')
    m.CCG = pyo.Param(m.rdn_plant, m.prgen, m.size, initialize = data['ccg'], doc = 'Capital cost of parallal gens. (M$/unit)')
    m.CCS = pyo.Param(initialize=data['CCS'], doc = 'Capital cost of storage (M$/unit)') 
    m.FOC_E = pyo.Param(m.old_plant, initialize = data['foc_e'], doc = 'Fixed operating cost of preinstalled plants (M$/unit)')
    m.FOC_P = pyo.Param(m.new_plant, m.size, initialize = data['foc_p'], doc = 'Fixed operating cost of potential plants (M$/unit)')     
    m.FOC_G = pyo.Param(m.rdn_plant, m.prgen, m.size, initialize = data['foc_g'], doc = 'Fixed operating cost of parallal gens. (M$/unit)')
    m.FOC_S = pyo.Param(initialize=data['FOC_S'], doc = 'Fixed operating cost of storage ') 
    m.VOC_P = pyo.Param(m.plant, initialize = data['voc_p'], doc = 'Variable operating cost of plants ($/MWh)')
    m.VOC_G  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['voc_g'], doc = 'Variable operating cost of parallel gens. ($/MWh)') 
    m.VOC_S = pyo.Param(m.res_plant, m.strge, initialize = data['voc_s'], doc = 'Variable operating cost of batteries ($/MWh)')
    m.EFF = pyo.Param(m.plant, initialize = data['eff'], doc = 'Power efficiency (MMBtu/MWh)')
    m.FPC = pyo.Param(m.plant, initialize = data['fpc'], doc = 'Fuel price ($/MMBtu)')    
    m.COE = pyo.Param(m.plant, initialize = data['coe'], doc = 'CO2 emission rate (kg/MMBtu)')
    m.UEC = pyo.Param(m.old_plant, initialize = data['uec'], doc = 'Plant extension cost (M$)')  
    m.RT = pyo.Param(m.rg_plt_ex, initialize = data['rt'], doc = 'Remaining lifetime (years)')  
    m.SUC_P = pyo.Param(m.dpt_plant, initialize = data['suc_p'], doc = 'Start-up cost of plants (M$)') 
    m.SDC_P = pyo.Param(m.dpt_plant, initialize = data['sdc_p'], doc = 'Shut-down cost of plants (M$)')
    m.SUC_G  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['suc_g'], doc = 'Start-up cost of parallel gens. (M$)')  
    m.SDC_G  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['sdc_g'], doc = 'Shut-down cost of parallel gens. (M$)')        
    m.RUP = pyo.Param(m.dpt_plant, initialize = data['rup'], doc = 'Ramping up ratio of plants (%)') 
    m.RDP = pyo.Param(m.dpt_plant, initialize = data['rdp'], doc = 'Ramping down ratio of plants (%)') 
    m.RUG  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['rug'], doc = 'Ramping up ratio of parallel gens. (%)')  
    m.RDG  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['rdg'], doc = 'Ramping down ratio of parallel gens. (%)')  
    m.DCF = pyo.Param(m.year, initialize = data['dcf'], doc = 'Discount factor')  
    m.CTX = pyo.Param(m.year, initialize = data['ctx'], doc = 'CO2 tax') 
    m.PKD = pyo.Param(m.year, initialize = data['pkd'], doc = 'Peak demand (MW)')  

    m.Prob_H1 = pyo.Param(m.state, m.rdn_plant, initialize = data['prob_H1'], doc='State probability of Design 1')    
    m.Prob_H2 = pyo.Param(m.state, m.rdn_plant, initialize = data['prob_H2'], doc='State probability of Design 2') 
    m.Prob_H3 = pyo.Param(m.state, m.rdn_plant, initialize = data['prob_H3'], doc='State probability of Design 3')          
    m.bigM_prod = pyo.Param(m.plant, initialize = data['bigM_prod'], doc = 'Upper bounds of expected production level')
    m.bigM_plant_cap = pyo.Param(m.plant, initialize = data['bigM_plant_cap'], doc = 'Upper bounds of plant capacity')  
    
    
    # Bounds for disjunctions
    def _bounds_cpa_rule(m, r, k, t):
        for k in m.plant:
            return (0, m.bigM_plant_cap[k])
    
    def _bounds_cpo_rule(m, r, k, t, n, b):
        for k in m.plant:
            return (0, m.bigM_plant_cap[k])
    
    def _bounds_cav_rule(m, r, k, j, t):
        for k in m.rdn_plant:
            return (0, m.CPG[k,j,3])

    def _bounds_cop_rule(m, r, k, j, t, n, b):
        for k in m.rdn_plant:
            return (0, m.CPG[k,j,3])    
    
    def _bounds_acf_rule(m, s, r, k, t, n, b):
        for k in m.plant:
            return (0, m.bigM_prod[k])
    
    def _bounds_esp_rule(m, r, k, t, n, b):
        for k in m.plant:
            return (0, m.bigM_prod[k])


    # Non-negative variables
    m.cpi = pyo.Var(m.rg_plt_pn, m.year, within = pyo.NonNegativeReals, doc = 'Installed capacity of plants')   
    m.cpa = pyo.Var(m.rg_plt, m.year, within = pyo.NonNegativeReals, bounds=_bounds_cpa_rule, doc = 'Available capacity of plants') 
    m.cpo = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, bounds=_bounds_cpo_rule, doc = 'Operating capacity of plants')
    m.cupo = pyo.Var(m.rg_plt, m.size, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Unit operating capacity of plants')
    m.cin = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within = pyo.NonNegativeReals, doc = 'Installed capacity of parallel gens.') 
    m.cav = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within = pyo.NonNegativeReals, bounds=_bounds_cav_rule, doc = 'Available capacity of parallel gens.') 
    m.cop = pyo.Var(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, bounds=_bounds_cop_rule, doc = 'Operating capacity of parallel gens.')
    m.cbop = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Unit operating capacity of parallel gens.')
    m.csi = pyo.Var(m.rg_plt_rw, m.strge, m.year, within = pyo.NonNegativeReals, doc = 'Installed capacity of batteries')
    m.csa = pyo.Var(m.rg_plt_rw, m.strge, m.year, within = pyo.NonNegativeReals, doc = 'Available capacity of batteries')
    m.ss = pyo.Var(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Storage level of batteries')
    m.chl = pyo.Var(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Charging level of batteries')
    m.dcl = pyo.Var(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Discharging level of batteries')
    m.tch = pyo.Var(m.rg_plt_rw, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Total charging level of batteries')
    m.tdc = pyo.Var(m.rg_plt_rw, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Total discharging level of batteries')
    
    m.rf = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.Reals, bounds=(0, 1), doc = 'Successful operational relibility')  
    m.rp = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.Reals, bounds=(0, 1), doc = 'Partial operational reliability')
    m.acf = pyo.Var(m.state, m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, bounds=_bounds_acf_rule, doc = 'Available capacity at failure state') 
    m.esp = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, bounds=_bounds_esp_rule, doc = 'Expected power production')
    m.fs = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Amount of feedstocks') 
    m.dt = pyo.Var(m.rg_plt, m.year, within = pyo.NonNegativeReals, doc = 'Downtime') 
    m.umd = pyo.Var(m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Unmet demand (load shedding)')
    m.ct = pyo.Var(m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, doc = 'Curtailment')  

    # Binary variables
    m.ypi = pyo.Var(m.rg_plt_pn, m.year, within=pyo.Binary, doc = 'Installation of plants')      
    m.yupi = pyo.Var(m.rg_plt_pn, m.size, m.year, within=pyo.Binary, doc = 'Unit installation of plants')
    m.ypl = pyo.Var(m.rg_plt_ex, m.year, within=pyo.Binary, doc = 'Lifetime extension of exsiting plants') 
    m.ypa = pyo.Var(m.rg_plt, m.year, within=pyo.Binary, doc = 'Availability of plants')  
    m.yupa = pyo.Var(m.rg_plt, m.size, m.year, within=pyo.Binary, doc = 'Unit availability of plants')  
    m.xpo = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Operation of plants')  
    m.xupo = pyo.Var(m.rg_plt, m.size, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Unit operation of plants') 
    m.usu = pyo.Var(m.rg_plt_dp, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Startup of plants') 
    m.usd = pyo.Var(m.rg_plt_dp, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Shutdown of plants') 
    m.uusu = pyo.Var(m.rg_plt_dp, m.size, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Unit startup of plants') 
    m.uusd = pyo.Var(m.rg_plt_dp, m.size, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Unit shutdown of plants')

    m.yin = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within=pyo.Binary, doc = 'Installation of parallel gens.') 
    m.ybin = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, within=pyo.Binary, doc = 'Unit installation of parallel gens.') 
    m.yav = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within=pyo.Binary, doc = 'Availability of parallel gens.') 
    m.ybav = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, within=pyo.Binary, doc = 'unit availability of parallel gens.') 
    m.xop = pyo.Var(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Operation of parallel gens.') 
    m.xbop = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Unit operation of parallel gens.')
    m.psu = pyo.Var(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Startup of parallel gens.')
    m.psd = pyo.Var(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Shutdown of parallel gens.') 
    m.ppsu = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Unit startup of parallel gens.') 
    m.ppsd = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Unit shutdown of parallel gens.')
    
    m.ysi = pyo.Var(m.rg_plt_rw, m.strge, m.year, within=pyo.Binary, doc = 'Installation of parallel batteries') 
    m.ysa = pyo.Var(m.rg_plt_rw, m.strge, m.year, within=pyo.Binary, doc = 'Availability of parallel batteries') 
    m.xso = pyo.Var(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Operation of parallel batteries') 
    m.och = pyo.Var(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Charge of parallel batteries') 
    m.odc = pyo.Var(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub, within=pyo.Binary, doc = 'Discharge of parallel batteries')    
    
    m.TT = pyo.Var(m.year, m.rpdn, m.sub, within=pyo.Binary)
    
    # Disaggregated variables
    m.esp1 = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, bounds=_bounds_esp_rule) 
    m.esp2 = pyo.Var(m.rg_plt, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals, bounds=_bounds_esp_rule)      
    m.umd1 = pyo.Var(m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)  
    m.umd2 = pyo.Var(m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)    
    m.ct1 = pyo.Var(m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)   
    m.ct2 = pyo.Var(m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)
    m.tch1 = pyo.Var(m.rg_plt_rw, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)
    m.tch2 = pyo.Var(m.rg_plt_rw, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)
    m.tdc1 = pyo.Var(m.rg_plt_rw, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)
    m.tdc2 = pyo.Var(m.rg_plt_rw, m.year, m.rpdn, m.sub, within = pyo.NonNegativeReals)
  
 
    ################################           Objective fnuction            ################################
    
    @m.Expression()
    def investment_cost_plant(m):
        return sum(m.DCF[t] * (sum(m.CCP[k,c] * m.yupi[r,k,c,t] for r,k in m.rg_plt_pn for c in m.size) +
                               sum(m.UEC[k] * m.ypl[r,k,t] for r,k in m.rg_plt_ex)) for t in m.year)
    
    @m.Expression()
    def investment_cost_parallel(m):
        return sum(m.DCF[t] * m.CCG[k,j,c] * m.ybin[r,k,j,c,t] for r,k in m.rg_plt_rn for j in m.prgen for c in m.size for t in m.year)
    
    @m.Expression()
    def investment_cost_storage(m):
        return sum(m.DCF[t] * m.CCS * m.ysi[r,k,i,t] for r,k in m.rg_plt_rw for i in m.strge for t in m.year)

    @m.Expression(doc="CAPEX")
    def capital_expenditure(m):
        return m.investment_cost_plant + m.investment_cost_parallel + m.investment_cost_storage

    @m.Expression()
    def fixed_operating_cost(m):
        return sum(m.DCF[t] * (sum(m.FOC_P[k,c] * m.yupa[r,k,c,t] for r,k in m.rg_plt_pn for c in m.size) +
                               sum(m.FOC_E[k] * m.ypa[r,k,t] for r,k in m.rg_plt_ex for t in m.year) + 
                               sum(m.FOC_G[k,j,c] * m.ybav[r,k,j,c,t] for r,k in m.rg_plt_rn for j in m.prgen for c in m.size) + 
                               sum(m.FOC_S * m.ysa[r,k,i,t] for r,k in m.rg_plt_rw for i in m.strge)) for t in m.year)
    
    @m.Expression()
    def variable_operating_cost(m):
        return sum(m.DCF[t] * m.OT[n,b] * (sum(m.VOC_P[k] * m.cpo[r,k,t,n,b] / 1000 for r,k in m.rg_plt) + 
                                           sum(m.VOC_G[k,j] * m.cop[r,k,j,t,n,b] / 1000 for r,k in m.rg_plt_rn for j in m.prgen) + 
                                           sum(m.VOC_S[k,i] * m.ss[r,k,i,t,n,b] /1000 for r,k in m.rg_plt_rw for i in m.strge)) 
                   for t in m.year for n in m.rpdn for b in m.sub)

    @m.Expression()
    def startup_cost(m):
        return sum(m.DCF[t] * (sum(m.SUC_P[k] * m.usu[r,k,t,n,b] for r,k in m.rg_plt_dp) +
                               sum(m.SUC_G[k,j] * m.psu[r,k,j,t,n,b] for r,k in m.rg_plt_rn for j in m.prgen)) 
                   for t in m.year for n in m.rpdn for b in m.sub)
     
    @m.Expression()
    def shutdown_cost(m):
        return sum(m.DCF[t] * (sum(m.SDC_P[k] * m.usd[r,k,t,n,b] for r,k in m.rg_plt_dp) + 
                               sum(m.SDC_G[k,j] * m.psd[r,k,j,t,n,b] for r,k in m.rg_plt_rn for j in m.prgen)) 
                   for t in m.year for n in m.rpdn for b in m.sub)
    
    @m.Expression()
    def fuel_cost(m):
        return sum(m.DCF[t] * m.FPC[k] * m.fs[r,k,t,n,b] for r,k in m.rg_plt_dp for t in m.year for n in m.rpdn for b in m.sub)
    
    @m.Expression()
    def emission_cost(m):
        return sum(m.DCF[t] * m.CTX[t] * m.COE[k] * m.fs[r,k,t,n,b] for r,k in m.rg_plt for t in m.year for n in m.rpdn for b in m.sub)     

    @m.Expression(doc="OPEX")
    def operating_expenses(m):
        return m.fixed_operating_cost + m.variable_operating_cost + m.startup_cost + m.shutdown_cost + m.fuel_cost + m.emission_cost

    @m.Expression()
    def downtime_penalty(m):
        return sum(m.DCF[t] * m.beta * m.dt[r,k,t] for r,k in m.rg_plt for t in m.year)     

    @m.Expression()
    def loadshedding_penalty(m):
        return sum(m.DCF[t] * m.alpha * m.umd[t,n,b] for r,k in m.rg_plt for t in m.year for n in m.rpdn for b in m.sub)     

    @m.Objective(sense=pyo.minimize)
    def obj(m):
        return m.capital_expenditure + m.operating_expenses + m.downtime_penalty + m.loadshedding_penalty       
    
    
    ################################               Constraints               ################################

    ###############          Installation and availability of potential power plants          ###############
    # Calculate installed capacity of power plants
    @m.Constraint(m.rg_plt_pn, m.year)
    def install_plant_cap(m, r, k, t):
        return m.cpi[r,k,t] == sum(m.CPP[k,c] * m.yupi[r,k,c,t] for c in m.size)

    # One capacity can be installed
    @m.Constraint(m.rg_plt_pn, m.year) 
    def install_plant_cap2(m, r, k, t):
        return m.ypi[r,k,t] == sum(m.yupi[r,k,c,t] for c in m.size)

    # The power plant can only be installed once over the horizon
    @m.Constraint(m.rg_plt_pn)
    def install_plant_limit(m, r, k):  
        return sum(m.ypi[r,k,t] for t in m.year) <= 1

    # Calculate available capacity of potential power plants
    @m.Constraint(m.rg_plt_pn, m.year) 
    def avail_ptplant_cap(m, r, k, t):
        return m.cpa[r,k,t] == sum(m.CPP[k,c] * m.yupa[r,k,c,t] for c in m.size)

    # One capacity can be available
    @m.Constraint(m.rg_plt_pn, m.year)
    def avail_ptplant_cap2(m, r, k, t):
        return m.ypa[r,k,t] == sum(m.yupa[r,k,c,t] for c in m.size)

    # Availability logics of potential power plants
    @m.Constraint(m.rg_plt_pn, m.year)
    def available_ptplant_logic(m, r, k, t):
        if t != 1:
            return m.ypa[r,k,t] == m.ypa[r,k,t-1] + m.ypi[r,k,t]
        else:
            return m.ypa[r,k,t] == m.ypi[r,k,t]


    #############         Availability and lifetime extension of existing power plants          #############
    # Calculate available capacity of existing power plants
    @m.Constraint(m.rg_plt_ex, m.year) 
    def avail_explant_cap(m, r, k, t):
        return m.cpa[r,k,t] == m.PRE[k] * m.ypa[r,k,t] 

    # Existing power plants should be extended to operate after their lifetimes
    @m.Constraint(m.rg_plt_ex, m.year)
    def lifetime_extension1(m, r, k, t):
        if t == m.RT[r,k] + 1:
            return m.ypa[r,k,t] == m.ypl[r,k,t]
        else:
            return pyo.Constraint.Skip

    # Availability logics for existing conventional power plants
    @m.Constraint(m.rg_plt_ex_cv, m.year)
    def available_ex_cv_plant_logic(m, r, k, t):
        if t != 1:
            return m.ypa[r,k,t] <= m.ypa[r,k,t-1]
        else:
            return m.ypa[r,k,t] <= 1
        
    # Availability logics for existing renewable power plants
    @m.Constraint(m.rg_plt_ex_re, m.year)
    def available_ex_res_plant_logic(m, r, k, t):
        if (t >= 1) and (t <= m.RT[r,k]):
            return m.ypa[r,k,t] == 1
        else:
            return pyo.Constraint.Skip        


    ###############                  Operation of dispatchable power plants                   ###############
    
    ###########                  Operation of dispatchable existing power plants                  ###########
    # Propositions of availability and operation
    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn, m.sub)
    def opt_ex_plant_logic1(m, r, k, t, n, b):
        return m.xpo[r,k,t,n,b] <= m.ypa[r,k,t] 
    
    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn, m.sub)
    def opt_ex_plant_logic2(m, r, k, t, n, b):
        return m.usu[r,k,t,n,b] <= m.ypa[r,k,t] 
    
    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn, m.sub)
    def opt_ex_plant_logic3(m, r, k, t, n, b):
        return m.usd[r,k,t,n,b] <= m.ypa[r,k,t] 
    
    
    # Operation limits
    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn, m.sub)   
    def plant_ex_opt1(m, r, k, t, n, b):
        return m.MINP[k] * m.PRE[k] * m.xpo[r,k,t,n,b] <= m.cpo[r,k,t,n,b]  
    
    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn, m.sub)
    def plant_ex_opt2(m, r, k, t, n, b):
        return m.cpo[r,k,t,n,b] <= m.MAXP[k] * m.PRE[k] * m.xpo[r,k,t,n,b]

    
    # Ramping up constraints (including time scale connection)
    @m.Constraint(m.rg_plt_dp_ex)
    def ex_rudp1(m, r, k):
        return m.cpo[r,k,1,1,1] <= m.PRE[k] * m.RUP[k] * m.usu[r,k,1,1,1]
    
    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn, m.sub)
    def ex_rudp2(m, r, k, t, n, b):
        if b != 1:
            return  m.cpo[r,k,t,n,b] - m.cpo[r,k,t,n,b-1] <= m.PRE[k] * m.RUP[k] * (m.xpo[r,k,t,n,b-1] + m.usu[r,k,t,n,b])
        else:
            return pyo.Constraint.Skip  

    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn)            
    def ex_rudp3(m, r, k, t, n):
        if n != 1:
            return  m.cpo[r,k,t,n,1] - m.cpo[r,k,t,n-1,4] <= m.PRE[k] * m.RUP[k] * (m.xpo[r,k,t,n-1,4] + m.usu[r,k,t,n,1])
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp_ex, m.year)    
    def ex_rudp4(m, r, k, t):
        if t != 1:
            return  m.cpo[r,k,t,1,1] - m.cpo[r,k,t-1,4,4] <= m.PRE[k] * m.RUP[k] * (m.xpo[r,k,t-1,4,4] + m.usu[r,k,t,1,1])
        else:
            return pyo.Constraint.Skip
    
    
    # Ramping down constraints (including time scale connection)
    @m.Constraint(m.rg_plt_dp_ex)
    def ex_rddp1(m, r, k):
            return -m.cpo[r,k,1,1,1] <= m.PRE[k] * m.RDP[k] * (m.xpo[r,k,1,1,1] + m.usd[r,k,1,1,1])

    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn, m.sub)    
    def ex_rddp2(m, r, k, t, n, b):
        if b != 1:
            return  m.cpo[r,k,t,n,b-1] - m.cpo[r,k,t,n,b] <= m.PRE[k] * m.RDP[k] * (m.xpo[r,k,t,n,b-1] + m.usd[r,k,t,n,b])
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp_ex, m.year, m.rpdn)            
    def ex_rddp3(m, r, k, t, n):
        if n != 1:
            return  m.cpo[r,k,t,n-1,4] - m.cpo[r,k,t,n,1] <= m.PRE[k] * m.RDP[k] * (m.xpo[r,k,t,n,1] + m.usd[r,k,t,n,1])
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp_ex, m.year)     
    def ex_rddp4(m, r, k, t):
        if t != 1:
            return  m.cpo[r,k,t-1,4,4] - m.cpo[r,k,t,1,1] <= m.PRE[k] * m.RDP[k] * (m.xpo[r,k,t,4,4] + m.usd[r,k,t,1,1])
        else:
            return pyo.Constraint.Skip
    
    
    
    ###########                  Operation of dispatchable potential power plants                 ###########       
    # Propositions of availability and operation
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn, m.sub)
    def opt_pn_plant_logic1(m, r, k, c, t, n, b):
        return m.xupo[r,k,c,t,n,b] <= m.yupa[r,k,c,t] 

    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn, m.sub)
    def opt_pn_plant_logic2(m, r, k, c, t, n, b):
        return m.uusu[r,k,c,t,n,b] <= m.yupa[r,k,c,t] 
 
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn, m.sub)
    def opt_pn_plant_logic3(m, r, k, c, t, n, b):
        return m.uusd[r,k,c,t,n,b] <= m.yupa[r,k,c,t] 
    
    
    # Operation limits
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn, m.sub)
    def plant_pn_opt1(m, r, k, c, t, n, b):
        return m.MINP[k] * m.CPP[k,c] * m.xupo[r,k,c,t,n,b] <= m.cupo[r,k,c,t,n,b]
   
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn, m.sub)
    def plant_pn_opt2(m, r, k, c, t, n, b):
        return m.cupo[r,k,c,t,n,b] <= m.MAXP[k] * m.CPP[k,c] * m.xupo[r,k,c,t,n,b]
    
    
    # Logics between variables
    @m.Constraint(m.rg_plt_dp_pn, m.year, m.rpdn, m.sub)
    def plant_pn_opt_logic1(m, r, k, t, n, b):
        return m.xpo[r,k,t,n,b] == sum(m.xupo[r,k,c,t,n,b] for c in m.size)
    
    @m.Constraint(m.rg_plt_dp_pn, m.year, m.rpdn, m.sub)
    def plant_pn_opt_logic2(m, r, k, t, n, b):
        return m.cpo[r,k,t,n,b] == sum(m.cupo[r,k,c,t,n,b] for c in m.size)
    
     
    # Ramping up constraints (including time scale connection)
    @m.Constraint(m.rg_plt_dp_pn, m.size)
    def pn_rudp1(m, r, k, c):
        return m.cupo[r,k,c,1,1,1] <= m.CPP[k,c] * m.RUP[k] * m.uusu[r,k,c,1,1,1]
    
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn, m.sub)
    def pn_rudp2(m, r, k, c, t, n, b):
        if b != 1:
            return  m.cupo[r,k,c,t,n,b] - m.cupo[r,k,c,t,n,b-1] <= m.CPP[k,c] * m.RUP[k] * (m.xupo[r,k,c,t,n,b-1] + m.uusu[r,k,c,t,n,b])
        else:
            return pyo.Constraint.Skip 
    
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn)        
    def pn_rudp3(m, r, k, c, t, n):
        if n != 1:
            return  m.cupo[r,k,c,t,n,1] - m.cupo[r,k,c,t,n-1,4] <= m.CPP[k,c] * m.RUP[k] * (m.xupo[r,k,c,t,n-1,4] + m.uusu[r,k,c,t,n,1])
        else:
            return pyo.Constraint.Skip
    
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year)
    def pn_rudp4(m, r, k, c, t):
        if t != 1:
            return  m.cupo[r,k,c,t,1,1] - m.cupo[r,k,c,t-1,4,4] <= m.CPP[k,c] * m.RUP[k] * (m.xupo[r,k,c,t-1,4,4] + m.uusu[r,k,c,t,1,1])
        else:
            return pyo.Constraint.Skip
    
    
    # Ramping down constraints (including time scale connection)
    @m.Constraint(m.rg_plt_dp_pn, m.size)
    def pn_rddp1(m, r, k, c):
            return -m.cupo[r,k,c,1,1,1] <= m.CPP[k,c] * m.RDP[k] * (m.xupo[r,k,c,1,1,1] + m.uusd[r,k,c,1,1,1])
    
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn, m.sub)
    def pn_rddp2(m, r, k, c, t, n, b):
        if b != 1:
            return  m.cupo[r,k,c,t,n,b-1] - m.cupo[r,k,c,t,n,b] <= m.CPP[k,c] * m.RDP[k] * (m.xupo[r,k,c,t,n,b-1] + m.uusd[r,k,c,t,n,b])
        else:
            return pyo.Constraint.Skip  
    
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year, m.rpdn)        
    def pn_rddp3(m, r, k, c, t, n):
        if n != 1:
            return  m.cupo[r,k,c,t,n-1,4] - m.cupo[r,k,c,t,n,1] <= m.CPP[k,c] * m.RDP[k] * (m.xupo[r,k,c,t,n,1] + m.uusd[r,k,c,t,n,1])
        else:
            return pyo.Constraint.Skip
    
    @m.Constraint(m.rg_plt_dp_pn, m.size, m.year)
    def pn_rddp4(m, r, k, c, t):
        if t != 1:
            return  m.cupo[r,k,c,t-1,4,4] - m.cupo[r,k,c,t,1,1] <= m.CPP[k,c] * m.RDP[k] * (m.xupo[r,k,c,t,4,4] + m.uusd[r,k,c,t,1,1])
        else:
            return pyo.Constraint.Skip  
    
    
    # Start-up/shut-down logics
    @m.Constraint(m.rg_plt_dp_pn, m.year, m.rpdn, m.sub)
    def stup_logic1(m, r, k, t, n, b):
        return m.usu[r,k,t,n,b] == sum(m.uusu[r,k,c,t,n,b] for c in m.size)
    
    @m.Constraint(m.rg_plt_dp_pn, m.year, m.rpdn, m.sub)
    def shdw_logic1(m, r, k, t, n, b):
        return m.usd[r,k,t,n,b] == sum(m.uusd[r,k,c,t,n,b] for c in m.size)

    
    
    ###########              Unit committment of dispatchable potential power plants             ###########             
    # Start-up logics
    @m.Constraint(m.rg_plt_dp)
    def stu_logic1(m, r, k):
        return m.xpo[r,k,1,1,1] <= m.usu[r,k,1,1,1]

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)    
    def stu_logic2(m, r, k, t, n, b):
        if b != 1:
            return  m.xpo[r,k,t,n,b] - m.xpo[r,k,t,n,b-1] <= m.usu[r,k,t,n,b]
        else:
            return pyo.Constraint.Skip 
    
    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn) 
    def stu_logic3(m, r, k, t, n):
        if n != 1:
            return  m.xpo[r,k,t,n,1] - m.xpo[r,k,t,n-1,4] <= m.usu[r,k,t,n,1]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year) 
    def stu_logic4(m, r, k, t):
        if t != 1:
            return  m.xpo[r,k,t,1,1] - m.xpo[r,k,t-1,4,4] <= m.usu[r,k,t,1,1]
        else:
            return pyo.Constraint.Skip


    @m.Constraint(m.rg_plt_dp)
    def stu_opt1(m, r, k):
        return m.usu[r,k,1,1,1] <= 1 
    
    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)
    def stu_opt2(m, r, k, t, n, b):
        if b != 1:
            return  m.usu[r,k,t,n,b] + m.xpo[r,k,t,n,b-1] <= 1
        else:
            return pyo.Constraint.Skip 
    
    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn)
    def stu_opt3(m, r, k, t, n):
        if n != 1:
            return  m.usu[r,k,t,n,1] + m.xpo[r,k,t,n-1,4] <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year)
    def stu_opt4(m, r, k, t):
        if t != 1:
            return  m.usu[r,k,t,1,1] + m.xpo[r,k,t-1,4,4] <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)
    def stu_opt5(m, r, k, t, n, b):
        return m.usu[r,k,t,n,b] <= m.xpo[r,k,t,n,b]


    # Shut-down logics
    @m.Constraint(m.rg_plt_dp)
    def shd_logic1(m, r, k):
        return -m.xpo[r,k,1,1,1] <= m.usd[r,k,1,1,1]

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)
    def shd_logic2(m, r, k, t, n, b):
        if b != 1:
            return  m.xpo[r,k,t,n,b-1] - m.xpo[r,k,t,n,b] <= m.usd[r,k,t,n,b]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn)
    def shd_logic3(m, r, k, t, n):
        if n != 1:
            return  m.xpo[r,k,t,n-1,4] - m.xpo[r,k,t,n,1] <= m.usd[r,k,t,n,1]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year)
    def shd_logic4(m, r, k, t):
        if t != 1:
            return  m.xpo[r,k,t-1,4,4] - m.xpo[r,k,t,1,1] <= m.usd[r,k,t,1,1]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)
    def shd_opt1(m, r, k, t, n, b):
        return m.usd[r,k,t,n,b] + m.xpo[r,k,t,n,b] <= 1

    @m.Constraint(m.rg_plt_dp)
    def shd_opt2(m, r, k):
        return m.usd[r,k,1,1,1] <= 0 

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)
    def shd_opt3(m, r, k, t, n, b):
        if b != 1:
            return  m.usd[r,k,t,n,b] <= m.xpo[r,k,t,n,b-1] 
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn)
    def shd_opt4(m, r, k, t, n):
        if n != 1:
            return  m.usd[r,k,t,n,1] <= m.xpo[r,k,t,n-1,4] 
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year)
    def shd_opt5(m, r, k, t):
        if t != 1:
            return  m.usd[r,k,t,1,1] <= m.xpo[r,k,t-1,4,4]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)
    def susd_logic(m, r, k, t, n, b):
        return m.usd[r,k,t,n,b] + m.usu[r,k,t,n,b] <= 1
    
    
    
    ###############                Operation of non-dispatchable power plants                 ###############
    # Operation limits
    @m.Constraint(m.rg_plt_rw, m.year, m.rpdn, m.sub)
    def plant_rw_opt1(m, r, k, t, n, b):
        return m.cpo[r,k,t,n,b] == m.CPF[k,t,n,b] * m.cpa[r,k,t]
   
    @m.Constraint(m.rg_plt_rw, m.year, m.rpdn, m.sub)
    def plant_rw_opt2(m, r, k, t, n, b):
        return m.ypa[r,k,t] == m.xpo[r,k,t,n,b] 
    
    
    
    ###############           Installation and availability of parallel generators            ###############
    # Calculate capacity of installed parallel generators
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def install_gen_cap(m, r, k, j, t):
        return m.cin[r,k,j,t] == sum(m.CPG[k,j,c] * m.ybin[r,k,j,c,t] for c in m.size)

    # One capacity can be installed
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year) 
    def install_gen_cap2(m, r, k, j, t):
        return m.yin[r,k,j,t] == sum(m.ybin[r,k,j,c,t] for c in m.size)

    # Parallel generators can only be installed once over the horizon
    @m.Constraint(m.rg_plt_rn, m.prgen)
    def install_gen_limit1(m, r, k, j):  
        return sum(m.yin[r,k,j,t] for t in m.year) <= 1

    # Calculate capacity of available parallel generators
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year) 
    def avail_gen_cap(m, r, k, j, t):
        return m.cav[r,k,j,t] == sum(m.CPG[k,j,c] * m.ybav[r,k,j,c,t] for c in m.size)

    # One capacity can be available
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year) 
    def avail_gen_cap2(m, r, k, j, t):
        return m.yav[r,k,j,t] == sum(m.ybav[r,k,j,c,t] for c in m.size)

    # Availability logic of parallel generators
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year)
    def available_gen_logic(m, r, k, j, c, t):
        if t != 1:
            return m.ybav[r,k,j,c,t] == m.ybav[r,k,j,c,t-1] + m.ybin[r,k,j,c,t]
        else:
            return m.ybav[r,k,j,c,t] == m.ybin[r,k,j,c,t]
     
    # Backup and power plant logics
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def back_main_ava(m, r, k, j, t):
        return m.yav[r,k,j,t] <= m.ypa[r,k,t]

            

    ###############                     Operation of parallel generators                      ###############
    # Logics
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub)
    def opt_gen_logic1(m, r, k, j, c, t, n, b):
        return m.xbop[r,k,j,c,t,n,b] <= m.ybav[r,k,j,c,t] 

    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub)    
    def opt_gen_logic2(m, r, k, j, c, t, n, b):
        return m.ppsu[r,k,j,c,t,n,b] <= m.ybav[r,k,j,c,t] 
  
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub)
    def opt_gen_logic3(m, r, k, j, c, t, n, b):
        return m.ppsd[r,k,j,c,t,n,b] <= m.ybav[r,k,j,c,t] 


    # Operation limits
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub)
    def gen_opt1(m, r, k, j, c, t, n, b):
        return m.MING[k,j] * m.CPG[k,j,c] * m.xbop[r,k,j,c,t,n,b] <= m.cbop[r,k,j,c,t,n,b] 

    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub)
    def gen_opt2(m, r, k, j, c, t, n, b):
        return m.cbop[r,k,j,c,t,n,b] <= m.MAXG[k,j] * m.CPG[k,j,c] * m.xbop[r,k,j,c,t,n,b]
    
        
    # Logics between variables
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def gen_opt_logic1(m, r, k, j, t, n, b):
        return m.xop[r,k,j,t,n,b] == sum(m.xbop[r,k,j,c,t,n,b] for c in m.size)

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def gen_opt_logic2(m, r, k, j, t, n, b):
        return m.cop[r,k,j,t,n,b] == sum(m.cbop[r,k,j,c,t,n,b] for c in m.size)

        
    # Ramping up constraints (including time scale connection)
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size)
    def rugen1(m, r, k, j, c):
        return m.cbop[r,k,j,c,1,1,1] <= m.CPG[k,j,c] * m.RUG[k,j] * m.ppsu[r,k,j,c,1,1,1]
    
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub)
    def rugen2(m, r, k, j, c, t, n, b):
        if b != 1:
            return  m.cbop[r,k,j,c,t,n,b] - m.cbop[r,k,j,c,t,n,b-1] <= m.CPG[k,j,c] * m.RUG[k,j] * (m.xbop[r,k,j,c,t,n,b-1] + m.ppsu[r,k,j,c,t,n,b])
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn)
    def rugen3(m, r, k, j, c, t, n):
        if n != 1:
            return  m.cbop[r,k,j,c,t,n,1] - m.cbop[r,k,j,c,t,n-1,4] <= m.CPG[k,j,c] * m.RUG[k,j] * (m.xbop[r,k,j,c,t,n-1,4] + m.ppsu[r,k,j,c,t,n,1])
        else:
            return pyo.Constraint.Skip
    
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year)
    def rugen4(m, r, k, j, c, t):
        if t != 1:
            return  m.cbop[r,k,j,c,t,1,1] - m.cbop[r,k,j,c,t-1,4,4] <= m.CPG[k,j,c] * m.RUG[k,j] * (m.xbop[r,k,j,c,t-1,4,4] + m.ppsu[r,k,j,c,t,1,1])
        else:
            return pyo.Constraint.Skip     
    
    
    # Ramping down constraints (including time scale connection)
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size)
    def rdgen1(m, r, k, j, c):
            return -m.cbop[r,k,j,c,1,1,1] <= m.CPG[k,j,c] * m.RDG[k,j] * (m.xbop[r,k,j,c,1,1,1] + m.ppsd[r,k,j,c,1,1,1])

    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn, m.sub)
    def rdgen2(m, r, k, j, c, t, n, b):
        if b != 1:
            return  m.cbop[r,k,j,c,t,n,b-1] - m.cbop[r,k,j,c,t,n,b] <= m.CPG[k,j,c] * m.RDG[k,j] * (m.xbop[r,k,j,c,t,n,b-1] + m.ppsd[r,k,j,c,t,n,b])
        else:
            return pyo.Constraint.Skip
    
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year, m.rpdn)        
    def rdgen3(m, r, k, j, c, t, n):
        if n != 1:
            return  m.cbop[r,k,j,c,t,n-1,4] - m.cbop[r,k,j,c,t,n,1] <= m.CPG[k,j,c] * m.RDG[k,j] * (m.xbop[r,k,j,c,t,n,1] + m.ppsd[r,k,j,c,t,n,1])
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year)
    def rdgen4(m, r, k, j, c, t):
        if t != 1:
            return  m.cbop[r,k,j,c,t-1,4,4] - m.cbop[r,k,j,c,t,1,1] <= m.CPG[k,j,c] * m.RDG[k,j] * (m.xbop[r,k,j,c,t,4,4] + m.ppsd[r,k,j,c,t,1,1])
        else:
            return pyo.Constraint.Skip



    ###########                      Unit commitment of parallel generators                     ###########
    # Start-up/shut-down logics
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def stup_gen_logic1(m, r, k, j, t, n, b):
        return m.psu[r,k,j,t,n,b] == sum(m.ppsu[r,k,j,c,t,n,b] for c in m.size)

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def shdw_gen_logic1(m, r, k, j, t, n, b):
        return m.psd[r,k,j,t,n,b] == sum(m.ppsd[r,k,j,c,t,n,b] for c in m.size)


    # Start-up constraints
    @m.Constraint(m.rg_plt_rn, m.prgen)
    def stu_gen_logic1(m, r, k, j):
        return m.xop[r,k,j,1,1,1] <= m.psu[r,k,j,1,1,1]

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def stu_gen_logic2(m, r, k, j, t, n, b):
        if b != 1:
            return  m.xop[r,k,j,t,n,b] - m.xop[r,k,j,t,n,b-1] <= m.psu[r,k,j,t,n,b]
        else:
            return pyo.Constraint.Skip
  
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn)
    def stu_gen_logic3(m, r, k, j, t, n):
        if n != 1:
            return  m.xop[r,k,j,t,n,1] - m.xop[r,k,j,t,n-1,4] <= m.psu[r,k,j,t,n,1]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def stu_gen_logic4(m, r, k, j, t):
        if t != 1:
            return  m.xop[r,k,j,t,1,1] - m.xop[r,k,j,t-1,4,4] <= m.psu[r,k,j,t,1,1]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen)
    def stu_gen_opt1(m, r, k, j):
        return m.psu[r,k,j,1,1,1] <= 1 

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def stu_gen_opt2(m, r, k, j, t, n, b):
        if b != 1:
            return  m.psu[r,k,j,t,n,b] + m.xop[r,k,j,t,n,b-1] <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn)
    def stu_gen_opt3(m, r, k, j, t, n):
        if n != 1:
            return  m.psu[r,k,j,t,n,1] + m.xop[r,k,j,t,n-1,4] <= 1
        else:
            return pyo.Constraint.Skip
 
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def stu_gen_opt4(m, r, k, j, t):
        if t != 1:
            return  m.psu[r,k,j,t,1,1] + m.xop[r,k,j,t-1,4,4] <= 1
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def stu_gen_opt5(m, r, k, j, t, n, b):
        return m.psu[r,k,j,t,n,b] <= m.xop[r,k,j,t,n,b]


    # Shut-down constraints
    @m.Constraint(m.rg_plt_rn, m.prgen)
    def shd_gen_logic1(m, r, k, j):
        return -m.xop[r,k,j,1,1,1] <= m.psd[r,k,j,1,1,1]

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def shd_gen_logic2(m, r, k, j, t, n, b):
        if b != 1:
            return  m.xop[r,k,j,t,n,b-1] - m.xop[r,k,j,t,n,b] <= m.psd[r,k,j,t,n,b]
        else:
            return pyo.Constraint.Skip
 
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn)
    def shd_gen_logic3(m, r, k, j, t, n):
        if n != 1:
            return  m.xop[r,k,j,t,n-1,4] - m.xop[r,k,j,t,n,1] <= m.psd[r,k,j,t,n,1]
        else:
            return pyo.Constraint.Skip
  
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def shd_gen_logic4(m, r, k, j, t):
        if t != 1:
            return  m.xop[r,k,j,t-1,4,4] - m.xop[r,k,j,t,1,1] <= m.psd[r,k,j,t,1,1]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def shd_gen_opt1(m, r, k, j, t, n, b):
        return m.psd[r,k,j,t,n,b] + m.xop[r,k,j,t,n,b] <= 1

    @m.Constraint(m.rg_plt_rn, m.prgen)
    def shd_gen_opt2(m, r, k, j):
        return m.psd[r,k,j,1,1,1] <= 0 

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def shd_gen_opt3(m, r, k, j, t, n, b):
        if b != 1:
            return  m.psd[r,k,j,t,n,b] <= m.xop[r,k,j,t,n,b-1] 
        else:
            return pyo.Constraint.Skip
  
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn)
    def shd_gen_opt4(m, r, k, j, t, n):
        if n != 1:
            return  m.psd[r,k,j,t,n,1] <= m.xop[r,k,j,t,n-1,4] 
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def shd_gen_opt5(m, r, k, j, t):
        if t != 1:
            return  m.psd[r,k,j,t,1,1] <= m.xop[r,k,j,t-1,4,4]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def susd_gen_logic(m, r, k, j, t, n, b):
        return m.psd[r,k,j,t,n,b] + m.psu[r,k,j,t,n,b] <= 1

    # Operation logic b/w the main and backups
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year, m.rpdn, m.sub)
    def plant_gen_opt(m, r, k, j, t, n, b):
        return m.xop[r,k,j,t,n,b] <= m.xpo[r,k,t,n,b]
    
    
    
    ###############                Installation and availability of batteries                 ###############
    # Calculate capacity of installed parallel batteries
    @m.Constraint(m.rg_plt_rw, m.strge, m.year)
    def install_bat_cap(m, r, k, i, t):
        return m.csi[r,k,i,t] == m.CPS * m.ysi[r,k,i,t]

    # Parallel batteries can only be installed once over the horizon
    @m.Constraint(m.rg_plt_rw, m.strge)
    def install_bat_limit1(m, r, k, i):  
        return sum(m.ysi[r,k,i,t] for t in m.year) <= 1

    # Calculate capacity of available parallel batteries
    @m.Constraint(m.rg_plt_rw, m.strge, m.year) 
    def avail_bat_cap(m, r, k, i, t):
        return m.csa[r,k,i,t] == m.CPS * m.ysa[r,k,i,t]

    # Availability logic of parallel batteries
    @m.Constraint(m.rg_plt_rw, m.strge, m.year)
    def available_bat_logic(m, r, k, i, t):
        if t != 1:
            return m.ysa[r,k,i,t] == m.ysa[r,k,i,t-1] + m.ysi[r,k,i,t]
        else:
            return m.ysa[r,k,i,t] == m.ysi[r,k,i,t]
     
    # Backup and power plant logics
    @m.Constraint(m.rg_plt_rw, m.strge, m.year)
    def bat_main_ava(m, r, k, i, t):
        return m.ysa[r,k,i,t] <= m.ypa[r,k,t]  
    
    
    
    ###############                           Storage level balance                           ###############
    # Operation limits
    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def bat_opt1(m, r, k, i, t, n, b):
        return m.MINS[k,i] * m.CPS * m.xso[r,k,i,t,n,b] <= m.ss[r,k,i,t,n,b] 

    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def bat_opt2(m, r, k, i, t, n, b):
        return m.ss[r,k,i,t,n,b] <= m.MAXS[k,i] * m.CPS * m.xso[r,k,i,t,n,b]
    
    
    # Charging limits
    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def bat_che1(m, r, k, i, t, n, b):
        return m.CMIN * m.CPS * m.och[r,k,i,t,n,b] <= m.chl[r,k,i,t,n,b] 

    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def bat_che2(m, r, k, i, t, n, b):
        return m.chl[r,k,i,t,n,b]  <= m.CMAX * m.CPS * m.och[r,k,i,t,n,b]   
    
    
    # Discharging limits
    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def bat_dch1(m, r, k, i, t, n, b):
        return m.DMIN * m.CPS * m.odc[r,k,i,t,n,b] <= m.dcl[r,k,i,t,n,b] 

    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def bat_dch2(m, r, k, i, t, n, b):
        return m.dcl[r,k,i,t,n,b]  <= m.DMAX * m.CPS * m.odc[r,k,i,t,n,b]       
    
    
    # Logics
    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def opt_bat_logic1(m, r, k, i, t, n, b):
        return m.xso[r,k,i,t,n,b] <= m.ysa[r,k,i,t] 

    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)    
    def opt_bat_logic2(m, r, k, i, t, n, b):
        return m.xso[r,k,i,t,n,b] <= m.xpo[r,k,t,n,b] 
  
    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def opt_bat_logic3(m, r, k, i, t, n, b):
        return m.odc[r,k,i,t,n,b] <= m.ysa[r,k,i,t]   
    
    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def opt_bat_logic4(m, r, k, i, t, n, b):
        return m.och[r,k,i,t,n,b] <= m.ysa[r,k,i,t]       
    
    
    # Storage level balance
    @m.Constraint(m.rg_plt_rw, m.strge)
    def sto_level1(m, r, k, i):
        return m.ss[r,k,i,1,1,1] == m.CHE * m.chl[r,k,i,1,1,1] - m.dcl[r,k,i,1,1,1] / m.DCE 

    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn, m.sub)
    def sto_level2(m, r, k, i, t, n, b):
        if b != 1:
            return  m.ss[r,k,i,t,n,b] == (1 - m.ls) * m.ss[r,k,i,t,n,b-1] + m.CHE * m.chl[r,k,i,t,n,b] - m.dcl[r,k,i,t,n,b] / m.DCE
        else:
            return pyo.Constraint.Skip
  
    @m.Constraint(m.rg_plt_rw, m.strge, m.year, m.rpdn)
    def sto_level3(m, r, k, i, t, n):
        if n != 1:
            return  m.ss[r,k,i,t,n,1] == (1 - m.ls) * m.ss[r,k,i,t,n-1,4] + m.CHE * m.chl[r,k,i,t,n,1] - m.dcl[r,k,i,t,n,1] / m.DCE
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rw, m.strge, m.year)
    def sto_level4(m, r, k, i, t):
        if t != 1:
            return  m.ss[r,k,i,t,1,1] == (1 - m.ls) * m.ss[r,k,i,t-1,4,4] + m.CHE * m.chl[r,k,i,t,1,1] - m.dcl[r,k,i,t,1,1] / m.DCE
        else:
            return pyo.Constraint.Skip   
    
    
    # Peak demand constraint
    @m.Constraint(m.year)
    def tot_cap(m, t):
        return sum(m.cpa[r,k,t] for r,k in m.rg_plt) + sum(m.cav[r,k,j,t] for r,k in m.rg_plt_rn for j in m.prgen) >= m.PKD[t]
 
    ###############                     Operational reliability estimation                    ###############  
    
    # Z0, Z1, Z2, Z3 Disjuncts for design
    # W0, W1, W2, W3 Disjuncts for operation
    
    # Z0 (nothing installed)
    @m.Disjunct(m.rg_plt_rn, m.year)
    def Z0_disjunct(outer_Z0, r, k, t):
        m = outer_Z0.model()
        
        @outer_Z0.Constraint(m.rpdn, m.sub)
        def rf_z0(outer_Z0, n, b):
            return m.rf[r,k,t,n,b] == 1
            
        @outer_Z0.Constraint(m.rpdn, m.sub, m.state)
        def acf_z0(outer_Z0, n, b, s):
            return m.acf[s,r,k,t,n,b] == 0
 
        @outer_Z0.Constraint(m.rpdn, m.sub)
        def cpo_z0(outer_Z0, n, b):
            return m.cpo[r,k,t,n,b] == 0
 
        @outer_Z0.Constraint(m.prgen, m.rpdn, m.sub)
        def cop_z0(outer_Z0, j, n, b):
            return m.cop[r,k,j,t,n,b] == 0
 
        @outer_Z0.Constraint()
        def cpa_z0(outer_Z0):
            return m.cpa[r,k,t] == 0
 
        @outer_Z0.Constraint(m.prgen)
        def cav_z0(outer_Z0, j):
            return m.cav[r,k,j,t] == 0
 
 
 
    # Z1 (one installed)
    @m.Disjunct(m.rg_plt_rn, m.year)
    def Z1_disjunct(outer_Z1, r, k, t):
        m = outer_Z1.model()
        
        # Z1W0 (nothing operated)
        @outer_Z1.Disjunct(m.rpdn, m.sub)
        def z1w0(inner_Z1W0, n, b):
            
            @inner_Z1W0.Constraint()
            def rf_z1w0(inner_Z1W0):
                return m.rf[r,k,t,n,b] == 1
            
            @inner_Z1W0.Constraint(m.state)
            def acf_z1w0(inner_Z1W0, s):
                return m.acf[s,r,k,t,n,b] == 0
            
            @inner_Z1W0.Constraint()
            def cpo_z1w0(inner_Z1W0):
                return m.cpo[r,k,t,n,b] == 0
    
            @inner_Z1W0.Constraint(m.prgen)
            def cop_z1w0(inner_Z1W0, j):
                return m.cop[r,k,j,t,n,b] == 0
    
            @inner_Z1W0.Constraint()
            def cpa_z1w0(inner_Z1W0):
                return m.cpa[r,k,t] == 0
    
            @inner_Z1W0.Constraint(m.prgen)
            def cav_z1w0(inner_Z1W0, j):
                return m.cav[r,k,j,t] == 0
        
        # Z1W1 (one operated)
        @outer_Z1.Disjunct(m.rpdn, m.sub)
        def z1w1(inner_Z1W1, n, b):
            
            @inner_Z1W1.Constraint()
            def rf_z1w1(inner_Z1W1):
                return m.rf[r,k,t,n,b] == m.Prob_H1[1,k]    
            
            @inner_Z1W1.Constraint()
            def acf_z1w1(inner_Z1W1):
                return m.acf[1,r,k,t,n,b] == m.Prob_H1[1,k] * m.cpo[r,k,t,n,b]       
        
            @inner_Z1W1.Constraint()
            def acf_z1w1_2(inner_Z1W1):
                return m.acf[2,r,k,t,n,b] + m.acf[3,r,k,t,n,b] + m.acf[4,r,k,t,n,b] +m.acf[5,r,k,t,n,b] + \
                       m.acf[6,r,k,t,n,b] + m.acf[7,r,k,t,n,b] + m.acf[8,r,k,t,n,b] == 0 
        
        # Declare disjunction between inner disjuncts in Z1
        @outer_Z1.Disjunction(m.rpdn, m.sub)
        def z1w0_or_z1w1(outer_Z1, n, b):
            return [outer_Z1.z1w0[n,b], outer_Z1.z1w1[n,b]] 
 
            
    # Z2 (two installed)
    @m.Disjunct(m.rg_plt_rn, m.year)
    def Z2_disjunct(outer_Z2, r, k, t):
        
        # Z2W0 (nothing operated))
        @outer_Z2.Disjunct(m.rpdn, m.sub)
        def z2w0(inner_Z2W0, n, b):
            
            @inner_Z2W0.Constraint()
            def rf_z2w0(inner_Z2W0):
                return m.rf[r,k,t,n,b] == 1
            
            @inner_Z2W0.Constraint()
            def acf_z2w0(inner_Z2W0):
                return sum(m.acf[s,r,k,t,n,b] for s in m.state) == 0   
     
            @inner_Z2W0.Constraint()
            def cpo_z2w0(inner_Z2W0):
                return m.cpo[r,k,t,n,b] == 0
    
            @inner_Z2W0.Constraint(m.prgen)
            def cop_z2w0(inner_Z2W0, j):
                return m.cop[r,k,j,t,n,b] == 0
    
            @inner_Z2W0.Constraint()
            def cpa_z2w0(inner_Z2W0):
                return m.cpa[r,k,t] == 0
    
            @inner_Z2W0.Constraint(m.prgen)
            def cav_z2w0(inner_Z2W0, j):
                return m.cav[r,k,j,t] == 0     
     
            
        # Z2W1 (one operated)
        @outer_Z2.Disjunct(m.rpdn, m.sub)
        def z2w1(inner_Z2W1, n, b):
            
            @inner_Z2W1.Constraint()
            def rf_z2w1(inner_Z2W1):
                return m.rf[r,k,t,n,b] == m.Prob_H2[1,k] + m.Prob_H2[2,k] + m.Prob_H2[3,k]    
            
            @inner_Z2W1.Constraint()
            def acf_z2w1_1(inner_Z2W1):
                return m.acf[1,r,k,t,n,b] == m.Prob_H2[1,k] * m.cpo[r,k,t,n,b]       
        
            @inner_Z2W1.Constraint()
            def acf_z2w1_2(inner_Z2W1):
                return m.acf[2,r,k,t,n,b] == m.Prob_H2[2,k] * m.cpo[r,k,t,n,b]           
        
            @inner_Z2W1.Constraint()
            def acf_z2w1_3(inner_Z2W1):
                return m.acf[3,r,k,t,n,b] == m.Prob_H2[3,k] * m.cpo[r,k,t,n,b]  
                   
            @inner_Z2W1.Constraint()
            def acf_z2w1_4(inner_Z2W1):
                return m.acf[4,r,k,t,n,b] + m.acf[5,r,k,t,n,b] + \
                       m.acf[6,r,k,t,n,b] + m.acf[7,r,k,t,n,b] + m.acf[8,r,k,t,n,b] == 0
       
       
        # Z2W2 (two operate)
        @outer_Z2.Disjunct(m.rpdn, m.sub)
        def z2w2(inner_Z2W2, n, b):
            
            @inner_Z2W2.Constraint()
            def rf_z2w2(inner_Z2W2):
                return m.rf[r,k,t,n,b] == m.Prob_H2[1,k]   
            
            @inner_Z2W2.Constraint()
            def acf_z2w2_1(inner_Z2W2):
                return m.acf[1,r,k,t,n,b] == m.Prob_H2[1,k] * (m.cpo[r,k,t,n,b] + m.cop[r,k,1,t,n,b])       
        
            @inner_Z2W2.Constraint()
            def acf_z2w2_2(inner_Z2W2):
                return m.acf[2,r,k,t,n,b] == m.Prob_H2[2,k] * m.cpa[r,k,t]           
        
            @inner_Z2W2.Constraint()
            def acf_z2w2_3(inner_Z2W2):
                return m.acf[3,r,k,t,n,b] == m.Prob_H2[3,k] * m.cav[r,k,1,t]  
                   
            @inner_Z2W2.Constraint()
            def acf_z2w2_4(inner_Z2W2):
                return m.acf[4,r,k,t,n,b] +m.acf[5,r,k,t,n,b] + \
                       m.acf[6,r,k,t,n,b] + m.acf[7,r,k,t,n,b] + m.acf[8,r,k,t,n,b] == 0     
       
        # Declare disjunction between inner disjuncts in Z2
        @outer_Z2.Disjunction(m.rpdn, m.sub)
        def z2w0_or_z2w1_or_z2w2(outer_Z2, n, b):
            return [outer_Z2.z2w0[n,b], outer_Z2.z2w1[n,b], outer_Z2.z2w2[n,b]]  
 
 
    # Z3 (three installed)
    @m.Disjunct(m.rg_plt_rn, m.year)
    def Z3_disjunct(outer_Z3, r, k, t):
        
        # Z3W0 (nothing operates)
        @outer_Z3.Disjunct(m.rpdn, m.sub)
        def z3w0(inner_Z3W0, n, b):
            
            @inner_Z3W0.Constraint()
            def rf_z3w0(inner_Z3W0):
                return m.rf[r,k,t,n,b] == 1
            
            @inner_Z3W0.Constraint()
            def acf_z3w0(inner_Z3W0):
                return sum(m.acf[s,r,k,t,n,b] for s in m.state) == 0   

            @inner_Z3W0.Constraint()
            def cpo_z3w0(inner_Z3W0):
                return m.cpo[r,k,t,n,b] == 0
    
            @inner_Z3W0.Constraint(m.prgen)
            def cop_z3w0(inner_Z3W0, j):
                return m.cop[r,k,j,t,n,b] == 0
    
            @inner_Z3W0.Constraint()
            def cpa_z3w0(inner_Z3W0):
                return m.cpa[r,k,t] == 0
    
            @inner_Z3W0.Constraint(m.prgen)
            def cav_z3w0(inner_Z3W0, j):
                return m.cav[r,k,j,t] == 0                 
            
            
        # Z3W1 (one operated)
        @outer_Z3.Disjunct(m.rpdn, m.sub)
        def z3w1(inner_Z3W1, n, b):
            
            @inner_Z3W1.Constraint()
            def rf_z3w1(inner_Z3W1):
                return m.rf[r,k,t,n,b] == m.Prob_H3[1,k] + m.Prob_H3[2,k] + m.Prob_H3[3,k] + m.Prob_H3[4,k] + m.Prob_H3[5,k] + m.Prob_H3[6,k] + m.Prob_H3[7,k]
                
            @inner_Z3W1.Constraint()
            def acf_z3w1_1(inner_Z3W1):
                return m.acf[1,r,k,t,n,b] == m.Prob_H3[1,k] * m.cpo[r,k,t,n,b]       
        
            @inner_Z3W1.Constraint()
            def acf_z3w1_2(inner_Z3W1):
                return m.acf[2,r,k,t,n,b] == m.Prob_H3[2,k] * m.cpo[r,k,t,n,b]           
        
            @inner_Z3W1.Constraint()
            def acf_z3w1_3(inner_Z3W1):
                return m.acf[3,r,k,t,n,b] == m.Prob_H3[3,k] * m.cpo[r,k,t,n,b]  
 
            @inner_Z3W1.Constraint()
            def acf_z3w1_4(inner_Z3W1):
                return m.acf[4,r,k,t,n,b] == m.Prob_H3[4,k] * m.cpo[r,k,t,n,b]  
            
            @inner_Z3W1.Constraint()
            def acf_z3w1_5(inner_Z3W1):
                return m.acf[5,r,k,t,n,b] == m.Prob_H3[5,k] * m.cpo[r,k,t,n,b]
            
            @inner_Z3W1.Constraint()
            def acf_z3w1_6(inner_Z3W1):
                return m.acf[6,r,k,t,n,b] == m.Prob_H3[6,k] * m.cpo[r,k,t,n,b]
              
            @inner_Z3W1.Constraint()
            def acf_z3w1_7(inner_Z3W1):
                return m.acf[7,r,k,t,n,b] == m.Prob_H3[7,k] * m.cpo[r,k,t,n,b]     
                   
            @inner_Z3W1.Constraint()
            def acf_z3w1_8(inner_Z3W1):
                return m.acf[8,r,k,t,n,b] == 0
     
     
        # Z3W2 (two operated)
        @outer_Z3.Disjunct(m.rpdn, m.sub)
        def z3w2(inner_Z3W2, n, b):
            
            @inner_Z3W2.Constraint()
            def rf_z3w2(inner_Z3W2):
                return m.rf[r,k,t,n,b] == m.Prob_H3[1,k] + m.Prob_H3[2,k] + m.Prob_H3[3,k] + m.Prob_H3[5,k] 
                
            @inner_Z3W2.Constraint()
            def acf_z3w2_1(inner_Z3W2):
                return m.acf[1,r,k,t,n,b] == m.Prob_H3[1,k] * (m.cpo[r,k,t,n,b] + m.cop[r,k,1,t,n,b])       
        
            @inner_Z3W2.Constraint()
            def acf_z3w2_2(inner_Z3W2):
                return m.acf[2,r,k,t,n,b] == m.Prob_H3[2,k] * (m.cpo[r,k,t,n,b] + m.cop[r,k,1,t,n,b])           
        
            @inner_Z3W2.Constraint()
            def acf_z3w2_3(inner_Z3W2):
                return m.acf[3,r,k,t,n,b] == m.Prob_H3[3,k] * (m.cpo[r,k,t,n,b] + m.cop[r,k,1,t,n,b])  
 
            @inner_Z3W2.Constraint()
            def acf_z3w2_4(inner_Z3W2):
                return m.acf[4,r,k,t,n,b] == m.Prob_H3[4,k] * m.cpa[r,k,t]  
            
            @inner_Z3W2.Constraint()
            def acf_z3w2_5(inner_Z3W2):
                return m.acf[5,r,k,t,n,b] == m.Prob_H3[5,k] * (m.cpo[r,k,t,n,b] + m.cop[r,k,1,t,n,b])
            
            @inner_Z3W2.Constraint()
            def acf_z3w2_6(inner_Z3W2):
                return m.acf[6,r,k,t,n,b] == m.Prob_H3[6,k] * m.cav[r,k,1,t]
              
            @inner_Z3W2.Constraint()
            def acf_z3w2_7(inner_Z3W2):
                return m.acf[7,r,k,t,n,b] == m.Prob_H3[7,k] * m.cav[r,k,2,t]     
                   
            @inner_Z3W2.Constraint()
            def acf_z3w2_8(inner_Z3W2):
                return m.acf[8,r,k,t,n,b] == 0        
       
       
        # Z3W3 (three operated)
        @outer_Z3.Disjunct(m.rpdn, m.sub)
        def z3w3(inner_Z3W3, n, b):
            
            @inner_Z3W3.Constraint()
            def rf_z3w3(inner_Z3W3):
                return m.rf[r,k,t,n,b] == m.Prob_H3[1,k] 
                
            @inner_Z3W3.Constraint()
            def acf_z3w3_1(inner_Z3W3):
                return m.acf[1,r,k,t,n,b] == m.Prob_H3[1,k] * (m.cpo[r,k,t,n,b] + m.cop[r,k,1,t,n,b] + m.cop[r,k,2,t,n,b])       
        
            @inner_Z3W3.Constraint()
            def acf_z3w3_2(inner_Z3W3):
                return m.acf[2,r,k,t,n,b] == m.Prob_H3[2,k] * (m.cpa[r,k,t] + m.cav[r,k,2,t])           
        
            @inner_Z3W3.Constraint()
            def acf_z3w3_3(inner_Z3W3):
                return m.acf[3,r,k,t,n,b] == m.Prob_H3[3,k] * (m.cpa[r,k,t] + m.cav[r,k,1,t])  
 
            @inner_Z3W3.Constraint()
            def acf_z3w3_4(inner_Z3W3):
                return m.acf[4,r,k,t,n,b] == m.Prob_H3[4,k] * m.cpa[r,k,t]  
            
            @inner_Z3W3.Constraint()
            def acf_z3w3_5(inner_Z3W3):
                return m.acf[5,r,k,t,n,b] == m.Prob_H3[5,k] * (m.cav[r,k,1,t] + m.cav[r,k,2,t])
            
            @inner_Z3W3.Constraint()
            def acf_z3w3_6(inner_Z3W3):
                return m.acf[6,r,k,t,n,b] == m.Prob_H3[6,k] * m.cav[r,k,1,t]
              
            @inner_Z3W3.Constraint()
            def acf_z3w3_7(inner_Z3W3):
                return m.acf[7,r,k,t,n,b] == m.Prob_H3[7,k] * m.cav[r,k,2,t]     
                   
            @inner_Z3W3.Constraint()
            def acf_z3w3_8(inner_Z3W3):
                return m.acf[8,r,k,t,n,b] == 0           
       
        # Declare disjunction between inner disjuncts in Z3
        @outer_Z3.Disjunction(m.rpdn, m.sub)
        def z3w0_or_z3w1_or_z3w2_or_z3w3(outer_Z3, n, b):
            return [outer_Z3.z3w0[n,b], outer_Z3.z3w1[n,b], outer_Z3.z3w2[n,b], outer_Z3.z3w3[n,b]]
            
    # Declare disjunction between outer disjuncts 
    @m.Disjunction(m.rg_plt_rn, m.year)
    def outer_disjunctions(m, r, k, t):
        return [m.Z0_disjunct[r,k,t], m.Z1_disjunct[r,k,t], m.Z2_disjunct[r,k,t], m.Z3_disjunct[r,k,t]]  
 
 
    ###############                 Logics between design/operation/reliability               ###############  
    @m.Constraint(m.rg_plt_rn, m.year)
    def no_logic1(m, r, k, t):
        return m.Z0_disjunct[r,k,t].binary_indicator_var + m.ypa[r,k,t] <= 1

    @m.Constraint(m.rg_plt_rn, m.year)
    def no_logic2(m, r, k, t):
        return m.Z0_disjunct[r,k,t].binary_indicator_var + m.yav[r,k,1,t] <= 1

    @m.Constraint(m.rg_plt_rn, m.year)
    def no_logic3(m, r, k, t):
        return m.Z0_disjunct[r,k,t].binary_indicator_var + m.yav[r,k,2,t] <= 1


    @m.Constraint(m.rg_plt_rn, m.year)
    def one_logic1(m, r, k, t):
        return m.Z1_disjunct[r,k,t].binary_indicator_var <= m.ypa[r,k,t]

    @m.Constraint(m.rg_plt_rn, m.year)
    def one_logic2(m, r, k, t):
        return m.Z1_disjunct[r,k,t].binary_indicator_var + m.yav[r,k,1,t] <= 1

    @m.Constraint(m.rg_plt_rn, m.year)
    def one_logic3(m, r, k, t):
        return m.Z1_disjunct[r,k,t].binary_indicator_var + m.yav[r,k,2,t] <= 1


    @m.Constraint(m.rg_plt_rn, m.year)
    def two_logic1(m, r, k, t):
        return m.Z2_disjunct[r,k,t].binary_indicator_var <= m.ypa[r,k,t]

    @m.Constraint(m.rg_plt_rn, m.year)
    def two_logic2(m, r, k, t):
        return m.Z2_disjunct[r,k,t].binary_indicator_var <= m.yav[r,k,1,t]

    @m.Constraint(m.rg_plt_rn, m.year)
    def two_logic3(m, r, k, t):
        return m.Z2_disjunct[r,k,t].binary_indicator_var + m.yav[r,k,2,t] <= 1


    @m.Constraint(m.rg_plt_rn, m.year)
    def three_logic1(m, r, k, t):
        return m.Z3_disjunct[r,k,t].binary_indicator_var <= m.ypa[r,k,t]

    @m.Constraint(m.rg_plt_rn, m.year)
    def three_logic2(m, r, k, t):
        return m.Z3_disjunct[r,k,t].binary_indicator_var <= m.yav[r,k,1,t]

    @m.Constraint(m.rg_plt_rn, m.year)
    def three_logic3(m, r, k, t):
        return m.Z3_disjunct[r,k,t].binary_indicator_var <= m.yav[r,k,2,t]



    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def one_no_opt_logic1(m, r, k, t, n, b):
        return m.Z1_disjunct[r,k,t].z1w0[n,b].binary_indicator_var + m.xpo[r,k,t,n,b] <= 1

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def one_no_opt_logic2(m, r, k, t, n, b):
        return m.Z1_disjunct[r,k,t].z1w0[n,b].binary_indicator_var + m.xop[r,k,1,t,n,b] <= 1    

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def one_no_opt_logic3(m, r, k, t, n, b):
        return m.Z1_disjunct[r,k,t].z1w0[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1   
 
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_no_opt_logic1(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w0[n,b].binary_indicator_var + m.xpo[r,k,t,n,b] <= 1
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_no_opt_logic2(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w0[n,b].binary_indicator_var + m.xop[r,k,1,t,n,b] <= 1    

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_no_opt_logic3(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w0[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1    
 
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_no_opt_logic1(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w0[n,b].binary_indicator_var + m.xpo[r,k,t,n,b] <= 1
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_no_opt_logic2(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w0[n,b].binary_indicator_var + m.xop[r,k,1,t,n,b] <= 1    

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_no_opt_logic3(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w0[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1     
 
 
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def one_one_opt_logic1(m, r, k, t, n, b):
        return m.Z1_disjunct[r,k,t].z1w1[n,b].binary_indicator_var <= m.xpo[r,k,t,n,b]
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def one_one_opt_logic2(m, r, k, t, n, b):
        return m.Z1_disjunct[r,k,t].z1w1[n,b].binary_indicator_var + m.xop[r,k,1,t,n,b] <= 1    

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def one_one_opt_logic3(m, r, k, t, n, b):
        return m.Z1_disjunct[r,k,t].z1w1[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1    
 
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_one_opt_logic1(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w1[n,b].binary_indicator_var <= m.xpo[r,k,t,n,b]
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_one_opt_logic2(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w1[n,b].binary_indicator_var + m.xop[r,k,1,t,n,b] <= 1    

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_one_opt_logic3(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w1[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1   

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_one_opt_logic1(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w1[n,b].binary_indicator_var <= m.xpo[r,k,t,n,b]
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_one_opt_logic2(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w1[n,b].binary_indicator_var + m.xop[r,k,1,t,n,b] <= 1    

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_one_opt_logic3(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w1[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1   



    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_two_opt_logic1(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w2[n,b].binary_indicator_var <= m.xpo[r,k,t,n,b]
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_two_opt_logic2(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w2[n,b].binary_indicator_var <= m.xop[r,k,1,t,n,b] 

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def two_two_opt_logic3(m, r, k, t, n, b):
        return m.Z2_disjunct[r,k,t].z2w2[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1   

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_two_opt_logic1(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w2[n,b].binary_indicator_var <= m.xpo[r,k,t,n,b]
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_two_opt_logic2(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w2[n,b].binary_indicator_var <= m.xop[r,k,1,t,n,b]   

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_two_opt_logic3(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w2[n,b].binary_indicator_var + m.xop[r,k,2,t,n,b] <= 1   



    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_three_opt_logic1(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w3[n,b].binary_indicator_var <= m.xpo[r,k,t,n,b]
    
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_three_opt_logic2(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w3[n,b].binary_indicator_var <= m.xop[r,k,1,t,n,b]   

    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def three_three_opt_logic3(m, r, k, t, n, b):
        return m.Z3_disjunct[r,k,t].z3w3[n,b].binary_indicator_var <= m.xop[r,k,2,t,n,b]   



    # Total expected power output of power plants with redundancy
    @m.Constraint(m.rg_plt_rn, m.year, m.rpdn, m.sub)
    def tote_rn(m, r, k, t, n, b):
        return m.esp[r,k,t,n,b] == m.OT[n,b] * sum(m.acf[s,r,k,t,n,b] for s in m.state)

    # Total expected power output of power plants without redundancy
    @m.Constraint(m.rg_plt_nd, m.year, m.rpdn, m.sub)
    def tote_nd(m, r, k, t, n, b):
        return m.esp[r,k,t,n,b] == m.OT[n,b] * m.URP[k] * m.cpo[r,k,t,n,b] 
 
    # Successful operational reliability of power plants without redundancy
    @m.Constraint(m.rg_plt_nd, m.year, m.rpdn, m.sub)
    def successful_reliability(m, r, k, t, n, b):
        return m.rf[r,k,t,n,b] == m.URP[k] + (1 - m.URP[k]) * (1 - m.xpo[r,k,t,n,b])
     
    # Partial operational reliability 
    @m.Constraint(m.rg_plt, m.year, m.rpdn, m.sub)
    def relib(m, r, k, t, n, b):
        return m.rp[r,k,t,n,b] == 1 - m.rf[r,k,t,n,b]    

    # Total discharging level
    @m.Constraint(m.rg_plt_rw, m.year, m.rpdn, m.sub)
    def total_discharging(m, r, k, t, n, b):
        return m.tdc[r,k,t,n,b] == m.OT[n,b] * sum(m.dcl[r,k,i,t,n,b] for i in m.strge)

    # Total charging level
    @m.Constraint(m.rg_plt_rw, m.year, m.rpdn, m.sub)
    def total_charging(m, r, k, t, n, b):
        return m.tch[r,k,t,n,b] == m.OT[n,b] * sum(m.chl[r,k,i,t,n,b] for i in m.strge)

    # Amount of feedstock consumed
    @m.Constraint(m.rg_plt_dp, m.year, m.rpdn, m.sub)
    def feedstock(m, r, k, t, n, b):
        return m.fs[r,k,t,n,b] == m.esp[r,k,t,n,b] * m.EFF[k]
    
    # Symmetry breaking constraints
    @m.Constraint(m.rg_plt_rn, m.prgen, m.size, m.year)
    def sym1(m, r, k, j, c, t):
        if j !=1:
            return m.ybin[r,k,j,c,t] <= m.ybin[r,k,j-1,c,t]
        else:
            return pyo.Constraint.Skip

    @m.Constraint(m.rg_plt_rw, m.strge, m.year)
    def sym2(m, r, k, i, t):
        if i != 1:
            return m.ysi[r,k,i,t] <= m.ysi[r,k,i-1,t]
        else:
            return pyo.Constraint.Skip
    
    
    # Downtime 
    @m.Constraint(m.rg_plt, m.year)
    def downtime(m, r, k, t):
        return m.dt[r,k,t] == sum(m.rp[r,k,t,n,b] for n in m.rpdn for b in m.sub) / 4 * 4 * 8760


    # Disaggregated variables (for penalty calculation using Hull reformulation)
    @m.Constraint(m.rg_plt, m.year, m.rpdn, m.sub)
    def var1(m, r, k, t, n, b):
        return m.esp[r,k,t,n,b] == m.esp1[r,k,t,n,b] + m.esp2[r,k,t,n,b]

    @m.Constraint(m.rg_plt_rw, m.year, m.rpdn, m.sub)
    def var2(m, r, k, t, n, b):
        return m.tdc[r,k,t,n,b] == m.tdc1[r,k,t,n,b] + m.tdc2[r,k,t,n,b]

    @m.Constraint(m.rg_plt_rw, m.year, m.rpdn, m.sub)
    def var3(m, r, k, t, n, b):
        return m.tch[r,k,t,n,b] == m.tch1[r,k,t,n,b] + m.tch2[r,k,t,n,b]

    @m.Constraint(m.year, m.rpdn, m.sub)
    def var4(m, t, n, b):
        return m.umd[t,n,b] == m.umd1[t,n,b] + m.umd2[t,n,b]

    @m.Constraint(m.year, m.rpdn, m.sub)
    def var5(m, t, n, b):
        return m.ct[t,n,b] == m.ct1[t,n,b] + m.ct2[t,n,b]


    ## Undet demand & Curtailments 
    @m.Constraint(m.year, m.rpdn, m.sub)
    def production(m, t, n, b):
        return sum(m.OT[n,b] * m.D[r,t,n,b] * m.TT[t,n,b] for r in m.region) + sum(m.tch1[r,k,t,n,b] for r,k in m.rg_plt_rw) \
            <= sum(m.esp1[r,k,t,n,b] for r,k in m.rg_plt) + sum(m.tdc1[r,k,t,n,b] for r,k in m.rg_plt_rw)
    
    @m.Constraint(m.year, m.rpdn, m.sub)
    def unmet_demand1(m, t, n, b):
        return m.umd1[t,n,b] == 0
    
    @m.Constraint(m.year, m.rpdn, m.sub)
    def curtailment1(m, t, n, b):
        return m.ct1[t,n,b] == sum(m.esp1[r,k,t,n,b] for r,k in m.rg_plt) - sum(m.OT[n,b] * m.D[r,t,n,b] * m.TT[t,n,b] for r in m.region)
    
    
    @m.Constraint(m.year, m.rpdn, m.sub)
    def requirement(m, t, n, b):
        return sum(m.esp2[r,k,t,n,b] for r,k in m.rg_plt) + sum(m.tdc2[r,k,t,n,b] for r,k in m.rg_plt_rw) \
            <= sum(m.OT[n,b] * m.D[r,t,n,b] * (1 - m.TT[t,n,b]) for r in m.region) + sum(m.tch2[r,k,t,n,b] for r,k in m.rg_plt_rw)    
        
    @m.Constraint(m.year, m.rpdn, m.sub)
    def unmet_demand2(m, t, n, b):
        return m.umd2[t,n,b] == sum(m.OT[n,b] * m.D[r,t,n,b] * (1 - m.TT[t,n,b]) for r in m.region) - sum(m.esp2[r,k,t,n,b] for r,k in m.rg_plt)
    
    @m.Constraint(m.year, m.rpdn, m.sub)
    def curtailment2(m, t, n, b):
        return m.ct2[t,n,b] == 0       
 
 
    pyo.TransformationFactory(transformation).apply_to(m)
 
    return m

if __name__ == "__main__":
    d = dataset()
    m = RGEP_model('gdp.hull', d)

opt = pyo.SolverFactory('gurobi')
results = opt.solve(m, tee=True)