__author__ = "Seolhee Cho"

# GDP RGEP (Reliability-constrainted Generation Expansion Planning) model
# Investment decisions only
# Operation cost and reliability are estimated to tighten the lower bound.

import pyomo.environ as pyo
from input_data import dataset

def planning_master_model(data):
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


    # Parameters
    m.alpha = pyo.Param(initialize=data['alpha'], doc = 'Unmet demand penalty')  
    m.beta = pyo.Param(initialize=data['beta'], doc = 'Downtime penalty')      
    m.ETP  = pyo.Param(m.year, initialize=data['ETP'], doc = 'Yearly operation time (hours)')  
    m.av_CPF = pyo.Param(m.res_plant, m.year, initialize = data['av_CPF'], doc = 'Average capacity factor (%)')
    m.AD = pyo.Param(m.region, m.year, initialize = data['AD'], doc = 'Average power demand (MWh/subperiod)')   
    m.PRE = pyo.Param(m.old_plant, initialize = data['pre'], doc = 'Preinstalled capacity (MW)')  
    m.URP = pyo.Param(m.plant, initialize = data['urp'], doc = 'Unit reliability (1/failure rate)')
    m.URG  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['urg'], doc = 'Unit reliability of parallel gens. (1/failure rate)')      
    m.CPP = pyo.Param(m.new_plant, m.size, initialize = data['cpp'], doc = 'Nameplate capacity of potential plants (MW)') 
    m.CPG = pyo.Param(m.rdn_plant, m.prgen, m.size, initialize = data['cpg'], doc = 'Nameplate capacity of parallel gens. (MW)')   
    m.MINP = pyo.Param(m.dpt_plant, initialize = data['minp'], doc = 'Minimum operation ratio of plants (%)') 
    m.MING = pyo.Param(m.rdn_plant, m.prgen, initialize = data['ming'], doc = 'Minimum operation ratio of parallel gens. (%)') 
    m.MAXP = pyo.Param(m.dpt_plant, initialize = data['maxp'], doc = 'Maximum operation ratio of plants (%)') 
    m.MAXG = pyo.Param(m.rdn_plant, m.prgen, initialize = data['maxg'], doc = 'Maximum operation ratio of parallel gens. (%)') 
    m.CCP = pyo.Param(m.new_plant, m.size, initialize = data['ccp'], doc = 'Capital cost of potential plants (M$/unit)')
    m.CCG = pyo.Param(m.rdn_plant, m.prgen, m.size, initialize = data['ccg'], doc = 'Capital cost of parallal gens. (M$/unit)')
    m.FOC_E = pyo.Param(m.old_plant, initialize = data['foc_e'], doc = 'Fixed operating cost of preinstalled plants (M$/unit)')
    m.FOC_P = pyo.Param(m.new_plant, m.size, initialize = data['foc_p'], doc = 'Fixed operating cost of potential plants (M$/unit)')     
    m.FOC_G = pyo.Param(m.rdn_plant, m.prgen, m.size, initialize = data['foc_g'], doc = 'Fixed operating cost of parallal gens. (M$/unit)')
    m.VOC_P = pyo.Param(m.plant, initialize = data['voc_p'], doc = 'Variable operating cost of plants ($/MWh)')
    m.VOC_G  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['voc_g'], doc = 'Variable operating cost of parallel gens. ($/MWh)') 
    m.EFF = pyo.Param(m.plant, initialize = data['eff'], doc = 'Power efficiency (MMBtu/MWh)')
    m.FPC = pyo.Param(m.plant, initialize = data['fpc'], doc = 'Fuel price ($/MMBtu)')    
    m.COE = pyo.Param(m.plant, initialize = data['coe'], doc = 'CO2 emission rate (kg/MMBtu)')
    m.UEC = pyo.Param(m.old_plant, initialize = data['uec'], doc = 'Plant extension cost (M$)')  
    m.RT = pyo.Param(m.rg_plt_ex, initialize = data['rt'], doc = 'Remaining lifetime (years)')  
    m.SUC_P = pyo.Param(m.dpt_plant, initialize = data['suc_p'], doc = 'Start-up cost of plants (M$)') 
    m.SDC_P = pyo.Param(m.dpt_plant, initialize = data['sdc_p'], doc = 'Shut-down cost of plants (M$)')
    m.SUC_G  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['suc_g'], doc = 'Start-up cost of parallel gens. (M$)')  
    m.SDC_G  = pyo.Param(m.rdn_plant, m.prgen, initialize = data['sdc_g'], doc = 'Shut-down cost of parallel gens. (M$)')        
    m.DCF = pyo.Param(m.year, initialize = data['dcf'], doc = 'Discount factor')  
    m.CTX = pyo.Param(m.year, initialize = data['ctx'], doc = 'CO2 tax') 
    m.PKD = pyo.Param(m.year, initialize = data['pkd'], doc = 'Peak demand (MW)')  



    # Non-negative variables
    m.cpi = pyo.Var(m.rg_plt_pn, m.year, within = pyo.NonNegativeReals, doc = 'Installed capacity of plants')   
    m.cpa = pyo.Var(m.rg_plt, m.year, within = pyo.NonNegativeReals, doc = 'Available capacity of plants') 
    m.cin = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within = pyo.NonNegativeReals, doc = 'Installed capacity of parallel gens.') 
    m.cav = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within = pyo.NonNegativeReals, doc = 'Available capacity of parallel gens.') 
    m.U_opt_p = pyo.Var(m.rg_plt, m.year, within = pyo.NonNegativeReals, doc = 'Estimated power level of main gens.')
    m.U_opt_b = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within = pyo.NonNegativeReals, doc = 'Estimated power level of parallel gens.')
    m.U_fs_p = pyo.Var(m.rg_plt, m.year, within = pyo.NonNegativeReals, doc = 'Estimated fuel consumption of main gens.')
    m.U_fs_b = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within = pyo.NonNegativeReals, doc = 'Estimated fuel consumption of parallel gens.')
    m.U_rf = pyo.Var(m.rg_plt, m.year, within = pyo.Reals, bounds=(0, 1), doc = 'Estimated successful operational relibility')  
    m.U_rp = pyo.Var(m.rg_plt, m.year, within = pyo.Reals, bounds=(0, 1), doc = 'Estimated partial operational reliability')
    m.U_dt = pyo.Var(m.rg_plt, m.year, within = pyo.NonNegativeReals, doc = 'Estimated downtime') 
    m.U_umd = pyo.Var(m.year, within = pyo.NonNegativeReals, doc = 'Estimated unmet demand (load shedding)')
    m.U_ct = pyo.Var(m.year, within = pyo.NonNegativeReals, doc = 'Estimated curtailment')  


    # Binary variables
    m.ypi = pyo.Var(m.rg_plt_pn, m.year, within=pyo.Binary, doc = 'Installation of plants')      
    m.yupi = pyo.Var(m.rg_plt_pn, m.size, m.year, within=pyo.Binary, doc = 'Unit installation of plants')
    m.ypl = pyo.Var(m.rg_plt_ex, m.year, within=pyo.Binary, doc = 'Lifetime extension of exsiting plants') 
    m.ypa = pyo.Var(m.rg_plt, m.year, within=pyo.Binary, doc = 'Availability of plants')  
    m.yupa = pyo.Var(m.rg_plt, m.size, m.year, within=pyo.Binary, doc = 'Unit availability of plants')  
    m.yin = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within=pyo.Binary, doc = 'Installation of parallel gens.') 
    m.ybin = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, within=pyo.Binary, doc = 'Unit installation of parallel gens.') 
    m.yav = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within=pyo.Binary, doc = 'Availability of parallel gens.') 
    m.ybav = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, within=pyo.Binary, doc = 'unit availability of parallel gens.')  

    m.U_yin = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within=pyo.Binary, doc = 'Master problem: Installation of parallel gens.') 
    m.U_ybin = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, within=pyo.Binary, doc = 'Master problem: Unit installation of parallel gens.') 
    m.U_yav = pyo.Var(m.rg_plt_rn, m.prgen, m.year, within=pyo.Binary, doc = 'Master problem: Availability of parallel gens.') 
    m.U_ybav = pyo.Var(m.rg_plt_rn, m.prgen, m.size, m.year, within=pyo.Binary, doc = 'Master problem: Unit availability of parallel gens.') 
    
    m.TT = pyo.Var(m.year, m.rpdn, m.sub, within=pyo.Binary)
    
    # Cost variables
    m.CAPEX = pyo.Var(within=pyo.NonNegativeReals)
    m.OPEX = pyo.Var(within=pyo.NonNegativeReals)
    m.FOC = pyo.Var(within=pyo.NonNegativeReals)
    m.VOC = pyo.Var(within=pyo.NonNegativeReals)
    m.SUC = pyo.Var(within=pyo.NonNegativeReals)
    m.SUC_E1 = pyo.Var(within=pyo.NonNegativeReals)
    m.SUC_E2 = pyo.Var(within=pyo.NonNegativeReals)
    m.SDC = pyo.Var(within=pyo.NonNegativeReals)
    m.SDC = pyo.Var(within=pyo.NonNegativeReals)
    m.FUC = pyo.Var(within=pyo.NonNegativeReals)  
    m.CEM = pyo.Var(within=pyo.NonNegativeReals)        
    m.DTP = pyo.Var(within=pyo.NonNegativeReals) 
    m.UMP = pyo.Var(within=pyo.NonNegativeReals) 
 
    ################################           Objective fnuction            ################################
    
    @m.Objective(sense=pyo.minimize)
    def obj(m):
        return m.CAPEX + m.OPEX + m.DTP + m.UMP

    @m.Constraint()
    def investment_cost(m):
        return m.CAPEX == sum(m.DCF[t] * (sum(m.CCP[k,c] * m.yupi[r,k,c,t] for r,k in m.rg_plt_pn for c in m.size) + 
                                          sum(m.UEC[k] * m.ypl[r,k,t] for r,k in m.rg_plt_ex) + 
                                          sum(m.CCG[k,j,c] * m.U_ybin[r,k,j,c,t] for r,k in m.rg_plt_rn for j in m.prgen for c in m.size)) for t in m.year)

    @m.Expression()
    def ICP(m):
        return sum(m.DCF[t] * (sum(m.CCP[k,c] * m.yupi[r,k,c,t] for r,k in m.rg_plt_pn for c in m.size) +
                               sum(m.UEC[k] * m.ypl[r,k,t] for r,k in m.rg_plt_ex)) for t in m.year)
    
    @m.Expression()
    def ICG(m):
        return sum(m.DCF[t] * m.CCG[k,j,c] * m.U_ybin[r,k,j,c,t] for r,k in m.rg_plt_rn for j in m.prgen for c in m.size for t in m.year)
    

    @m.Constraint()
    def Operating_expenses(m):
        return m.OPEX == m.FOC + m.VOC + m.SUC + m.SUC_E1 + m.SUC_E2 + m.SDC + m.FUC + m.CEM
        
    @m.Constraint()
    def fixed_operating_expenses(m):
        return m.FOC == sum(m.DCF[t] * (sum(m.FOC_P[k,c] * m.yupa[r,k,c,t] for r,k in m.rg_plt_pn for c in m.size) +
                                        sum(m.FOC_E[k] * m.ypa[r,k,t] for r,k in m.rg_plt_ex for t in m.year) + 
                                        sum(m.FOC_G[k,j,c] * m.U_ybav[r,k,j,c,t] for r,k in m.rg_plt_rn for j in m.prgen for c in m.size)) for t in m.year)
    
    @m.Constraint()
    def variable_operating_expenses(m):
        return m.VOC == sum(m.DCF[t] * (sum(m.VOC_P[k] * m.U_opt_p[r,k,t] for r,k in m.rg_plt) + 
                                        sum(m.VOC_G[k,j] * m.U_opt_b[r,k,j,t] for r,k in m.rg_plt_rn for j in m.prgen)) for t in m.year)

    @m.Constraint()
    def startup_cost(m):
        return m.SUC == sum(m.DCF[t] * m.SUC_P[k] * m.ypa[r,k,t] for r,k in m.rg_plt_rn for t in m.year) + \
            sum(m.SUC_G[k,j] * m.U_yav[r,k,j,t] for r,k in m.rg_plt_rn for j in m.prgen for t in m.year)
     
    @m.Constraint()
    def startup_cost_nuclear1(m):
        return m.SUC_E1 == m.DCF[1] * m.SUC_P['nuclear-old'] * m.ypa['r1','nuclear-old',1]
        
    @m.Constraint()
    def startup_cost_nuclear2(m):
        return m.SUC_E1 == m.DCF[1] * m.SUC_P['nuclear-new'] * m.ypa['r2','nuclear-new',1]
        
     
    @m.Constraint()
    def shutdown_cost(m):
        return m.SDC == 0
    
    @m.Constraint()
    def fuelcost(m):
        return m.FUC == sum(m.DCF[t] * (sum(m.FPC[k] * m.U_fs_p[r,k,t] for r,k in m.rg_plt) + sum(m.FPC[k] * m.U_fs_b[r,k,j,t] for r,k in m.rg_plt_rn for j in m.prgen)) 
                            for t in m.year)
    
    @m.Constraint()
    def emission_cost(m):
        return m.CEM == sum(m.DCF[t] * m.CTX[t] * (sum(m.COE[k] * m.U_fs_p[r,k,t] for r,k in m.rg_plt) + sum(m.COE[k] * m.U_fs_b[r,k,j,t] for r,k in m.rg_plt_rn for j in m.prgen))  
                                                   for t in m.year)     
 
    @m.Constraint()
    def downtime_penalty(m):
        return m.DTP == sum(m.DCF[t] * m.beta * m.U_dt[r,k,t] for r,k in m.rg_plt for t in m.year)     

    @m.Constraint()
    def emission_cost(m):
        return m.UMP == sum(m.DCF[t] * m.alpha * m.U_umd[t] for t in m.year)     
   
    
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
    
    
    ###############           Installation and availability of parallel generators            ###############
    # Calculate capacity of installed parallel generators
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def install_gen_cap(m, r, k, j, t):
        return m.cin[r,k,j,t] == sum(m.CPG[k,j,c] * m.U_ybin[r,k,j,c,t] for c in m.size)

    # One capacity can be installed
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year) 
    def install_gen_cap2(m, r, k, j, t):
        return m.U_yin[r,k,j,t] == sum(m.U_ybin[r,k,j,c,t] for c in m.size)

    # Parallel generators can only be installed once over the horizon
    @m.Constraint(m.rg_plt_rn, m.prgen)
    def install_gen_limit1(m, r, k, j):  
        return sum(m.U_yin[r,k,j,t] for t in m.year) <= 1

    # Calculate capacity of available parallel generators
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year) 
    def avail_gen_cap(m, r, k, j, t):
        return m.cav[r,k,j,t] == sum(m.CPG[k,j,c] * m.U_ybav[r,k,j,c,t] for c in m.size)

    # One capacity can be available
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year) 
    def avail_gen_cap2(m, r, k, j, t):
        return m.U_yav[r,k,j,t] == sum(m.U_ybav[r,k,j,c,t] for c in m.size)

    # Availability logic of parallel generators
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def available_gen_logic(m, r, k, j, t):
        if t != 1:
            return m.U_yav[r,k,j,t] == m.U_yav[r,k,j,t-1] + m.U_yin[r,k,j,t]
        else:
            return m.U_yav[r,k,j,t] == m.U_yin[r,k,j,t]
     
    # Backup and power plant logics
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def back_main_ava(m, r, k, j, t):
        return m.U_yav[r,k,j,t] <= m.ypa[r,k,t]

    # Symmetry breaking constraints
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def sym1(m, r, k, j, t):
        if j !=1:
            return m.U_yin[r,k,j,t] <= m.U_yin[r,k,j-1,t]
        else:
            return pyo.Constraint.Skip            

   # Peak demand constraint
    @m.Constraint(m.year)
    def tot_cap(m, t):
        return sum(m.cpa[r,k,t] for r,k in m.rg_plt) + sum(m.cav[r,k,j,t] for r,k in m.rg_plt_rn for j in m.prgen) >= m.PKD[t]


    #############################################              Operation estimation                 #############################################

    # Estimated operation level of main generator 
    @m.Constraint(m.rg_plt_rw, m.year)
    def opt_rw(m, r, k, t):
        return m.U_opt_p[r,k,t] == m.ETP[t] * m.av_CPF[k,t] * m.URP[k] * m.cpa[r,k,t]

    # Estimated operation level of main generator 
    @m.Constraint(m.rg_plt_dp, m.year)
    def opt_dp1(m, r, k, t):
        return m.U_opt_p[r,k,t] >= m.MINP[k] * m.ETP[t]  * m.URP[k] * m.cpa[r,k,t]
 
    # Estimated operation level of main generator 
    @m.Constraint(m.rg_plt_dp, m.year)
    def opt_dp2(m, r, k, t):
        return m.U_opt_p[r,k,t] <= m.MAXP[k] * m.ETP[t]  * m.URP[k] * m.cpa[r,k,t]
    
    # Estimated operation level of parallel generator 
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def opt_bk1(m, r, k, j, t):
        return m.U_opt_b[r,k,j,t] >= m.MING[k,j] * m.ETP[t]  * m.URG[k,j] * m.cav[r,k,j,t]
    
    # Estimated operation level of parallel generator 
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def opt_bk2(m, r, k, j, t):
        return m.U_opt_b[r,k,j,t] <= m.MAXG[k,j] * m.ETP[t]  * m.URG[k,j] * m.cav[r,k,j,t]    
    
    # Average demand satisfaction
    @m.Constraint(m.year)
    def demand(m, t):
        return  sum(m.U_opt_p[r,k,t] for r,k in m.rg_plt) + sum(m.U_opt_b[r,k,j,t] for r,k in m.rg_plt_rn for j in m.prgen) + m.U_umd[t] == \
            m.U_ct[t] + m.ETP[t] * sum(m.AD[r,t] for r in m.region)
       
    # Successful operational reliability of power plants without redundancy
    @m.Constraint(m.rg_plt_nd, m.year)
    def successful_reliability1(m, r, k, t):
        return m.U_rf[r,k,t] == m.URP[k] + (1 - m.URP[k]) * (1 - m.ypa[r,k,t])
            
    @m.Constraint(m.rg_plt_rn, m.year)
    def successful_reliability2(m, r, k, t):
        return m.U_rf[r,k,t] == m.URP[k] + (1 - m.URP[k]) * (1 - m.ypa[r,k,t]) + (1 - m.URP[k]) * m.URG[k,1] * m.U_yav[r,k,1,t] + \
            (1 - m.URP[k]) * (1 - m.URG[k,1]) * m.URG[k,2] * m.U_yav[r,k,2,t]
      
    # Downtime 
    @m.Constraint(m.rg_plt, m.year)
    def downtime(m, r, k, t):
        return m.U_dt[r,k,t] == (1 - m.U_rf[r,k,t]) * m.ETP[t]

    # Amount of feedstock consumed by main generators
    @m.Constraint(m.rg_plt, m.year)
    def feedstock1(m, r, k, t):
        return m.U_fs_p[r,k,t] == m.U_opt_p[r,k,t] * m.EFF[k]
    
    # Amount of feedstock consumed by parallel generators
    @m.Constraint(m.rg_plt_rn, m.prgen, m.year)
    def feedstock2(m, r, k, j, t):
        return m.U_fs_b[r,k,j,t] == m.U_opt_b[r,k,j,t] * m.EFF[k]

    return m


if __name__ == "__main__":
    d = dataset()
    m = planning_master_model(d)

opt = pyo.SolverFactory('gurobi')
results = opt.solve(m, tee=True)