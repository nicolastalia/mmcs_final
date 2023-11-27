import xpress as xp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def data_all():
    customer = pd.read_excel('231013_Customer_Base.xlsx')
    fraud = pd.read_excel('231013_Fraud_Cases.xlsx')
    transact = pd.read_excel('231013_Transactions_Input.xlsx')


    customer = pd.read_excel('231013_Customer_Base.xlsx')
    fraud = pd.read_excel('231013_Fraud_Cases.xlsx')
    transact = pd.read_excel('231013_Transactions_Input.xlsx')

    customer2 = pd.read_excel('231114_Customer_base_2nd_batch.xlsx')
    fraud2 = pd.read_excel('231114_Fraud_cases_2nd_batch.xlsx')
    transact2 = pd.read_excel('231114_Transactions_input_2nd_batch.xlsx')

    customer2.columns = ['customer_id', 'home_bank', 'customer_prob']
    customer = pd.concat([customer,customer2])
    # customer.drop_duplicates

    fraud = pd.concat([fraud,fraud2])
    # fraud.drop_duplicates

    transact = pd.concat([transact,transact2])
    # transact.drop_duplicates


    # Merging Transaction dataset with Customer dataset by customer_id
    full1 = transact.merge(customer, how = 'left', on='customer_id')
    full = full1.merge(fraud, how = 'left', on='transaction_id')

    # Remove all cash withdrawals, and all "paid_in" data from the dataset
    cleaned = full[(full.category != "Cash Withdrawal") & (full.In_or_Out != "paid_in")]
    cleaned['date'] = pd.to_datetime(cleaned['date'])

    # Subsetting to only the first day
    dayone = cleaned
    dayone = dayone.reset_index(drop = True) # reset the index, so the indexing is 0, 1, 2, ... 
    # drop = False would have made the old index into a new column in the dataframe. We don't need that, so drop = True.
    # Resetting the indexing is needed so that the index of the decision variable corresponds to the index of the dataframe; this is useful for subsequent analysis
    dayone['interbank'] = np.where((dayone['bank_to'] == dayone['bank_from']) | (dayone['bank_to'] == 'Intrnl'), 0, 1)
    dayone['day_of_month'] = dayone['date'].dt.day
    dayone['is_scam_transaction'] = dayone['is_scam_transaction'].fillna(0)
    dayone = dayone.drop_duplicates()
    dayone = dayone.reset_index(drop = True)
    return dayone

def first_phase(data_base,date_from,date_to,parameter):
    data = data_base[(data_base['date'] <= date_to) & (data_base['date'] >= date_from)]
    data.reset_index(inplace=True)
    banks = sorted(data['bank_from'].unique())
    priority = sorted(data['priority'].unique())
    category = sorted(data['category'].unique())
    day_month = sorted(data['day_of_month'].unique())
    bank_index = {}
    priority_index = {}
    category_index = {}
    day_month_index = {}
    for i in range(len(banks)):
        bank_index[banks[i]]= i

    for i in range(len(priority)):
        priority_index[priority[i]]= i

    for i in range(len(category)):
        category_index[category[i]]= i

    for i in range(len(day_month)):
        day_month_index[day_month[i]]= i

    prob = xp.problem('reg')

    # Decision variable

    banks_weight = np.array([xp.var(vartype=xp.continuous, name='weight_{0}'.format(banks[i]))
                        for i in range(len(banks))], dtype=xp.npvar)
    priority_weight = np.array([xp.var(vartype=xp.continuous, name='weight_priority_{0}'.format(priority[i]))
                        for i in range(len(priority))], dtype=xp.npvar)
    category_weight = np.array([xp.var(vartype=xp.continuous, name='weight_{0}'.format(category[i]))
                        for i in range(len(category))], dtype=xp.npvar)
    day_month_weight = np.array([xp.var(vartype=xp.continuous, name='weight_day_{0}'.format(day_month[i]))
                        for i in range(len(day_month))], dtype=xp.npvar)
    constant = xp.var(vartype=xp.continuous, name='constant')
    transac_prob_weight = xp.var(vartype=xp.continuous, name='transac_prob_weight')
    description_prob_weight = xp.var(vartype=xp.continuous, name='description_prob_weight')
    customer_prob_weight = xp.var(vartype=xp.continuous, name='customer_prob_weight')

    prob.addVariable(banks_weight,priority_weight,category_weight,day_month_weight,constant,transac_prob_weight,description_prob_weight,customer_prob_weight)

        
    prob.setObjective(
        xp.Sum((data.loc[c,'is_scam_transaction'] - 
        (constant + banks_weight[bank_index[data.loc[c,'home_bank']]]
        + priority_weight[priority_index[data.loc[c,'priority']]]
        + category_weight[category_index[data.loc[c,'category']]]
        + day_month_weight[day_month_index[data.loc[c,'day_of_month']]]
        + transac_prob_weight*data.loc[c,'transac_prob']
        + description_prob_weight*data.loc[c,'description_prob']
        + customer_prob_weight*data.loc[c,'customer_prob']
        ))**2
        for c in range(data.shape[0]))
        ,sense = xp.minimize)

    prob.write('reg','lp')

    prob.solve()

    print(f'First optimization ok with objective function value is {prob.getObjVal()}') 
    


    to_time_datetime = datetime.strptime(date_to, '%Y-%m-%d')
    new_date = to_time_datetime + timedelta(days=1)
    time_test = new_date.strftime('%Y-%m-%d')
    
    data_test = data_base[(data_base['date'] >= time_test) & (data_base['date'] <= time_test)].copy()
    data_test.reset_index(inplace=True)
    data_test['day_of_month'] = data_test['date'].dt.day
    print('data test imported without problems')

    for i in range(data_test.shape[0]):
        data_test.loc[i,'weight_bank'] = prob.getSolution(banks_weight[bank_index[data_test.loc[i,'home_bank']]])
        data_test.loc[i,'weight_priority'] = prob.getSolution(priority_weight[priority_index[data_test.loc[i,'priority']]])
        data_test.loc[i,'weight_category'] = prob.getSolution(category_weight[category_index[data_test.loc[i,'category']]])
        data_test.loc[i,'weight_day'] = prob.getSolution(day_month_weight[day_month_index[data_test.loc[i,'day_of_month']]])
        data_test.loc[i,'transac_prob_weight'] = prob.getSolution(transac_prob_weight)*data_test.loc[i,'transac_prob']
        data_test.loc[i,'description_prob_weight'] = prob.getSolution(description_prob_weight)*data_test.loc[i,'description_prob']
        data_test.loc[i,'customer_prob_weight'] = prob.getSolution(customer_prob_weight)*data_test.loc[i,'customer_prob']
        data_test.loc[i,'prediction'] = round(
            prob.getSolution(constant) 
            + data_test.loc[i,'weight_bank'] 
            + data_test.loc[i,'weight_priority'] 
            + data_test.loc[i,'weight_category'] 
            + data_test.loc[i,'weight_day']
            + data_test.loc[i,'transac_prob_weight']
            + data_test.loc[i,'description_prob_weight']
            + data_test.loc[i,'customer_prob_weight']
            ,4)
        if data_test.loc[i,'prediction'] > parameter:
            data_test.loc[i,'to_investigate'] = 1.0
        else:
            data_test.loc[i,'to_investigate'] = 0.0


    # Data
    team_size = {'bank_A':8,'bank_B':12,'bank_C':10,'bank_D':10,'bank_E':10 }
    bank_names = ['bank_A', 'bank_B', 'bank_C', 'bank_D', 'bank_E','Intrnl']
    time = {1: 0.25, 2:0.5, 3:1,4:2}
    ext_cost = {1: 40, 2:60, 3:100,4:150}

    # Problem set-up
    allocation_prob = xp.problem('allocation')

    # Decision Variable
    investigate = {}
    external = {}
    z = {}
    for i in range(len(data_test)):
        if data_test.loc[i,'interbank']==0:
            name = 'transaction_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])
            external_name = 'external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])
            investigate[name] = xp.var(vartype=xp.binary, name=name)
            external[external_name] = xp.var(vartype=xp.binary, name=external_name)
        elif data_test.loc[i,'interbank']==1:
            z_name = 'z_{0}'.format(data_test.loc[i,'transaction_id'])
            z[z_name] = xp.var(vartype=xp.binary, name=z_name)
            for j in ['bank_from','bank_to']:
                name = 'transaction_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,j])
                external_name = 'external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,j])
                investigate[name] = xp.var(vartype=xp.binary, name=name)
                external[external_name] = xp.var(vartype=xp.binary, name=external_name)
                

    allocation_prob.addVariable(investigate,external,z)

    # Constraints
    for key, value in team_size.items():
        allocation_prob.addConstraint(
            sum(time[data_test.loc[i,"priority"]] * 
                (investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,"bank_from"])] 
                - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])
                for i in range(len(data_test)) 
                if data_test.loc[i,'interbank']==0 and data_test.loc[i,'bank_from']==key)
            + sum(time[data_test.loc[i,"priority"]] * 0.5 * 
                (investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,j])] 
                - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,j])])
                for i in range(len(data_test)) for j in ['bank_from','bank_to'] 
                if data_test.loc[i,'interbank']==1 and data_test.loc[i,j]==key)
            <= value
        )

    for i in range(len(data_test)):
        if data_test.loc[i,'interbank']==1:
            allocation_prob.addConstraint(
                sum(
                    investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,j])] for j in ['bank_from','bank_to']
                ) == 2 * z['z_{0}'.format(data_test.loc[i,'transaction_id'])]
            )
            allocation_prob.addConstraint(
                external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])] == external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_to'])]
            )
            for k in ['bank_from','bank_to']:
                allocation_prob.addConstraint(
                    external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,k])] <= investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,k])]
                )
        elif data_test.loc[i,'interbank']==0:
            allocation_prob.addConstraint(
                external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])] <= investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,'bank_from'])])
        
    # Objective function: minimising the expected loss

    allocation_prob.setObjective(
        xp.Sum(data_test.loc[i,'Amount']*data_test.loc[i,'transac_prob']*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,"bank_from"])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])]*ext_cost[data_test.loc[i,'priority']]
                for i in range(len(data_test)) if (data_test.loc[i,"interbank"] == 0) and (data_test.loc[i,"prediction"] < 0.8))
        + xp.Sum(data_test.loc[i,'Amount']*data_test.loc[i,'transac_prob']*0.5*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,k])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,k])]*ext_cost[data_test.loc[i,'priority']]*0.5
                for i in range(len(data_test)) for k in ['bank_to','bank_from'] if (data_test.loc[i,"interbank"] == 1) and (data_test.loc[i,"prediction"] < 0.8) )
        + xp.Sum(data_test.loc[i,'Amount']*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,"bank_from"])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])]*ext_cost[data_test.loc[i,'priority']]
                for i in range(len(data_test)) if (data_test.loc[i,"interbank"] == 0) and (data_test.loc[i,"prediction"] >= 0.8) )
        + xp.Sum(data_test.loc[i,'Amount']*0.5*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,k])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,k])]*ext_cost[data_test.loc[i,'priority']]*0.5
                for i in range(len(data_test)) for k in ['bank_to','bank_from'] if (data_test.loc[i,"interbank"] == 1) and (data_test.loc[i,"prediction"] >= 0.8))
        , sense = xp.maximize)

    allocation_prob.write('test','lp')

    allocation_prob.solve()
    print(f'Second optimization ok with objective function value is {allocation_prob.getObjVal()}') 

    for i in range(data_test.shape[0]):
        data_test.loc[i,'chosen'] = allocation_prob.getSolution(investigate['transaction_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])
        data_test.loc[i,'ext_investigator'] = allocation_prob.getSolution(external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])
        data_test.loc[i,'cash_inv'] = allocation_prob.getSolution(external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])*ext_cost[data_test.loc[i,'priority']]
    return data_test



def retroactive(data_base,date_from,date_to,parameter):
    data = data_base[(data_base['date'] >= '2024-08-01') & (data_base['date'] <= '2024-12-31')]
    data.reset_index(inplace=True)
    banks = sorted(data['bank_from'].unique())
    priority = sorted(data['priority'].unique())
    category = sorted(data['category'].unique())
    day_month = sorted(data['day_of_month'].unique())
    bank_index = {}
    priority_index = {}
    category_index = {}
    day_month_index = {}
    for i in range(len(banks)):
        bank_index[banks[i]]= i

    for i in range(len(priority)):
        priority_index[priority[i]]= i

    for i in range(len(category)):
        category_index[category[i]]= i

    for i in range(len(day_month)):
        day_month_index[day_month[i]]= i

    prob = xp.problem('reg')

    # Decision variable

    banks_weight = np.array([xp.var(vartype=xp.continuous, name='weight_{0}'.format(banks[i]))
                        for i in range(len(banks))], dtype=xp.npvar)
    priority_weight = np.array([xp.var(vartype=xp.continuous, name='weight_priority_{0}'.format(priority[i]))
                        for i in range(len(priority))], dtype=xp.npvar)
    category_weight = np.array([xp.var(vartype=xp.continuous, name='weight_{0}'.format(category[i]))
                        for i in range(len(category))], dtype=xp.npvar)
    day_month_weight = np.array([xp.var(vartype=xp.continuous, name='weight_day_{0}'.format(day_month[i]))
                        for i in range(len(day_month))], dtype=xp.npvar)
    constant = xp.var(vartype=xp.continuous, name='constant')
    transac_prob_weight = xp.var(vartype=xp.continuous, name='transac_prob_weight')
    description_prob_weight = xp.var(vartype=xp.continuous, name='description_prob_weight')
    customer_prob_weight = xp.var(vartype=xp.continuous, name='customer_prob_weight')

    prob.addVariable(banks_weight,priority_weight,category_weight,day_month_weight,constant,transac_prob_weight,description_prob_weight,customer_prob_weight)

        
    prob.setObjective(
        xp.Sum((data.loc[c,'is_scam_transaction'] - 
        (constant + banks_weight[bank_index[data.loc[c,'home_bank']]]
        + priority_weight[priority_index[data.loc[c,'priority']]]
        + category_weight[category_index[data.loc[c,'category']]]
        + day_month_weight[day_month_index[data.loc[c,'day_of_month']]]
        + transac_prob_weight*data.loc[c,'transac_prob']
        + description_prob_weight*data.loc[c,'description_prob']
        + customer_prob_weight*data.loc[c,'customer_prob']
        ))**2
        for c in range(data.shape[0]))
        ,sense = xp.minimize)

    prob.write('reg','lp')

    prob.solve()

    print(f'First optimization ok with objective function value is {prob.getObjVal()}') 
    
    procesed = pd.DataFrame(columns=['index', 'transaction_id', 'description', 'Amount', 'category', 'date',
       'month', 'customer_id', 'type', 'In_or_Out', 'bank_to', 'bank_from',
       'transac_prob', 'description_prob', 'priority', 'home_bank',
       'customer_prob', 'is_scam_transaction', 'fraud_type', 'case_id',
       'interbank', 'day_of_month', 'weight_bank', 'weight_priority',
       'weight_category', 'weight_day', 'transac_prob_weight',
       'description_prob_weight', 'customer_prob_weight', 'prediction',
       'to_investigate', 'chosen', 'cash_inv'])

    to_time_datetime = datetime.strptime(date_to, '%Y-%m-%d')
    new_date = to_time_datetime + timedelta(days=1)
    time_test = new_date.strftime('%Y-%m-%d')
    
    data_test1 = data_base.copy()
    data_test1.reset_index(inplace=True)
    data_test1['day_of_month'] = data_test['date'].dt.day
    print('data test imported without problems')

    for i in range(data_test1.shape[0]):
        data_test1.loc[i,'weight_bank'] = prob.getSolution(banks_weight[bank_index[data_test.loc[i,'home_bank']]])
        data_test1.loc[i,'weight_priority'] = prob.getSolution(priority_weight[priority_index[data_test.loc[i,'priority']]])
        data_test1.loc[i,'weight_category'] = prob.getSolution(category_weight[category_index[data_test.loc[i,'category']]])
        data_test1.loc[i,'weight_day'] = prob.getSolution(day_month_weight[day_month_index[data_test.loc[i,'day_of_month']]])
        data_test1.loc[i,'transac_prob_weight'] = prob.getSolution(transac_prob_weight)*data_test.loc[i,'transac_prob']
        data_test1.loc[i,'description_prob_weight'] = prob.getSolution(description_prob_weight)*data_test1.loc[i,'description_prob']
        data_test1.loc[i,'customer_prob_weight'] = prob.getSolution(customer_prob_weight)*data_test1.loc[i,'customer_prob']
        data_test1.loc[i,'prediction'] = round(
            prob.getSolution(constant) 
            + data_test1.loc[i,'weight_bank'] 
            + data_test1.loc[i,'weight_priority'] 
            + data_test1.loc[i,'weight_category'] 
            + data_test1.loc[i,'weight_day']
            + data_test1.loc[i,'transac_prob_weight']
            + data_test1.loc[i,'description_prob_weight']
            + data_test1.loc[i,'customer_prob_weight']
            ,4)
        if data_test1.loc[i,'prediction'] > parameter:
            data_test1.loc[i,'to_investigate'] = 1.0
        else:
            data_test1.loc[i,'to_investigate'] = 0.0

        first_date = '2024-08-01'
        first_date = datetime.strptime(first_date, '%Y-%m-%d')
    for i in range(153):
        date = first_date + timedelta(days=i)
        data_test = data_test1[(data_test1['date'] >= date) & (data_test1['date'] <= date)].copy()
        data_test.reset_index(inplace=True)

        # Data
        team_size = {'bank_A':8,'bank_B':12,'bank_C':10,'bank_D':10,'bank_E':10 }
        bank_names = ['bank_A', 'bank_B', 'bank_C', 'bank_D', 'bank_E','Intrnl']
        time = {1: 0.25, 2:0.5, 3:1,4:2}
        ext_cost = {1: 40, 2:60, 3:100,4:150}

        # Problem set-up
        allocation_prob = xp.problem('allocation')

        # Decision Variable
        investigate = {}
        external = {}
        z = {}
        for i in range(len(data_test)):
            if data_test.loc[i,'interbank']==0:
                name = 'transaction_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])
                external_name = 'external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])
                investigate[name] = xp.var(vartype=xp.binary, name=name)
                external[external_name] = xp.var(vartype=xp.binary, name=external_name)
            elif data_test.loc[i,'interbank']==1:
                z_name = 'z_{0}'.format(data_test.loc[i,'transaction_id'])
                z[z_name] = xp.var(vartype=xp.binary, name=z_name)
                for j in ['bank_from','bank_to']:
                    name = 'transaction_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,j])
                    external_name = 'external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,j])
                    investigate[name] = xp.var(vartype=xp.binary, name=name)
                    external[external_name] = xp.var(vartype=xp.binary, name=external_name)
                    

        allocation_prob.addVariable(investigate,external,z)

        # Constraints
        for key, value in team_size.items():
            allocation_prob.addConstraint(
                sum(time[data_test.loc[i,"priority"]] * 
                    (investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,"bank_from"])] 
                    - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])
                    for i in range(len(data_test)) 
                    if data_test.loc[i,'interbank']==0 and data_test.loc[i,'bank_from']==key)
                + sum(time[data_test.loc[i,"priority"]] * 0.5 * 
                    (investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,j])] 
                    - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,j])])
                    for i in range(len(data_test)) for j in ['bank_from','bank_to'] 
                    if data_test.loc[i,'interbank']==1 and data_test.loc[i,j]==key)
                <= value
            )

        for i in range(len(data_test)):
            if data_test.loc[i,'interbank']==1:
                allocation_prob.addConstraint(
                    sum(
                        investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,j])] for j in ['bank_from','bank_to']
                    ) == 2 * z['z_{0}'.format(data_test.loc[i,'transaction_id'])]
                )
                allocation_prob.addConstraint(
                    external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])] == external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_to'])]
                )
                for k in ['bank_from','bank_to']:
                    allocation_prob.addConstraint(
                        external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,k])] <= investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,k])]
                    )
            elif data_test.loc[i,'interbank']==0:
                allocation_prob.addConstraint(
                    external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])] <= investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,'bank_from'])])
            
        # Objective function: minimising the expected loss

        allocation_prob.setObjective(
            xp.Sum(data_test.loc[i,'Amount']*data_test.loc[i,'transac_prob']*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,"bank_from"])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])]*ext_cost[data_test.loc[i,'priority']]
                    for i in range(len(data_test)) if (data_test.loc[i,"interbank"] == 0) and (data_test.loc[i,"prediction"] < 0.8))
            + xp.Sum(data_test.loc[i,'Amount']*data_test.loc[i,'transac_prob']*0.5*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,k])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,k])]*ext_cost[data_test.loc[i,'priority']]*0.5
                    for i in range(len(data_test)) for k in ['bank_to','bank_from'] if (data_test.loc[i,"interbank"] == 1) and (data_test.loc[i,"prediction"] < 0.8) )
            + xp.Sum(data_test.loc[i,'Amount']*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,"bank_from"])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])]*ext_cost[data_test.loc[i,'priority']]
                    for i in range(len(data_test)) if (data_test.loc[i,"interbank"] == 0) and (data_test.loc[i,"prediction"] >= 0.8) )
            + xp.Sum(data_test.loc[i,'Amount']*0.5*data_test.loc[i,'to_investigate']*investigate['transaction_{0}_{1}'.format(data_test.loc[i,"transaction_id"],data_test.loc[i,k])] - external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,k])]*ext_cost[data_test.loc[i,'priority']]*0.5
                    for i in range(len(data_test)) for k in ['bank_to','bank_from'] if (data_test.loc[i,"interbank"] == 1) and (data_test.loc[i,"prediction"] >= 0.8))
            , sense = xp.maximize)

        allocation_prob.write('test','lp')

        allocation_prob.solve()
        print(f'Second optimization ok with objective function value is {allocation_prob.getObjVal()}') 

        for i in range(data_test.shape[0]):
            data_test.loc[i,'chosen'] = allocation_prob.getSolution(investigate['transaction_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])
            data_test.loc[i,'ext_investigator'] = allocation_prob.getSolution(external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])
            data_test.loc[i,'cash_inv'] = allocation_prob.getSolution(external['external_{0}_{1}'.format(data_test.loc[i,'transaction_id'],data_test.loc[i,'bank_from'])])*ext_cost[data_test.loc[i,'priority']]
        procesed = pd.concat([procesed, data_test])
        print(f'iteration {i}')
    return procesed