#### Read Inputs #### 

import xpress as xp
import pandas as pd
import numpy as np
import random
import math
import time as tm

start = tm.time()

# pd.set_option('display.max_columns', None)
pd.options.mode.chained_assignment = None  # default='warn'. Uncomment this to remove warning messages.

print("Reading data...")
customer = pd.read_excel('231013_Customer_Base.xlsx')
fraud = pd.read_excel('231013_Fraud_Cases.xlsx')
transact = pd.read_excel('231013_Transactions_Input.xlsx')
customer2 = pd.read_excel('231114_Customer_Base_2nd_batch.xlsx')
customer2 = customer2.rename(columns={"bank_to" : "home_bank"}) # Rename column name 'bank_to' to 'home_bank' to be consistent
fraud2 = pd.read_excel('231114_Fraud_Cases_2nd_batch.xlsx')
transact2 = pd.read_excel('231114_Transactions_input_2nd_batch.xlsx')


#### Data Cleaning ####
print("Data cleaning...")
# Merging Transaction dataset with Customer dataset by customer_id
full = transact.merge(customer, how = 'left', on='customer_id')
# Remove all cash withdrawals, and all "paid_in" data from the dataset
cleaned = full[(full.category != "Cash Withdrawal") & (full.In_or_Out != "paid_in")]
cleaned["is_scam"] = cleaned["transaction_id"].isin(fraud["transaction_id"]) # Making boolean of True or False if its in the fraud database
cleaned["is_scam"] = cleaned["is_scam"].astype(int) # Converting boolean into 0 and 1

one_hot = pd.get_dummies(cleaned['category']) # Turn the categories into dummies
one_hot = one_hot.astype(int) # Change dummies into integers
full = cleaned.join(one_hot) # Add Dummies to full dataset
full = full.reset_index(drop = True) # Reset index of full dataset

full["day"] = (full["date"] - full["date"][0]).dt.days # "day" indicates which day

diction = {1: 0.25, # To convert priority into time spent investigating
           2: 0.5,
           3: 1,
           4: 2
}

full["time_spent"] = full["priority"] # Temporary
full = full.replace({"time_spent":diction}) # Replace priority with time spent using dictionary

# Drop streaming services from the dataset

####################################################################################################################


# Merging Transaction dataset with Customer dataset by customer_id
full2 = transact2.merge(customer2, how = 'left', on='customer_id')
# Remove all cash withdrawals, and all "paid_in" data from the dataset
cleaned2 = full2[(full2.category != "Cash Withdrawal") & (full2.In_or_Out != "paid_in")]
cleaned2["is_scam"] = cleaned2["transaction_id"].isin(fraud2["transaction_id"])
cleaned2["is_scam"] = cleaned2["is_scam"].astype(int)
one_hot2 = pd.get_dummies(cleaned2['category'])
one_hot2 = one_hot2.astype(int)
full2 = cleaned2.join(one_hot2)
full2 = full2.reset_index(drop = True)

full2["day"] = (full2["date"] - full2["date"][0]).dt.days

n_categories2 = np.shape(one_hot2)[1]
Categories2 = range(n_categories2)

full2["time_spent"] = full2["priority"]
full2 = full2.replace({"time_spent":diction})

n_daily_cases2 = np.shape(full2)[0]

#### NOTE! No Streaming Services category in dataset 2. Hence create a column of 0s in full2
full2["Streaming Services"] = 0
cols = full2.columns.tolist()
cols = cols[0:33] + cols[-1:] + cols[33:-1]
cols
full2 = full2[cols]

# Now we want to find the time taken to investigate any given case, based on their priority and based on whether it is shared.
temp = np.zeros((np.shape(full)[0], 6))
temp = pd.DataFrame(temp, columns = ['bank_A', 'bank_B', 'bank_C', 'bank_D', 'bank_E', 'Intrnl']) 
bank_to_df = temp.reset_index()
bank_to_df.update(full.pivot(columns = 'bank_to', values = 'time_spent')) # Pivot to get the time spent for each bank_to
bank_to_df = bank_to_df.set_index('index').rename_axis(None)
bank_to_df

bank_from_df = temp.reset_index()
bank_from_df.update(full.pivot(columns = 'bank_from', values = 'time_spent')) # Pivot to get the time spent for each bank_from
bank_from_df = bank_from_df.set_index('index').rename_axis(None)

ultimate = bank_to_df/2 + bank_from_df/2 # Halve and add them: this forces the 50:50 split for sharing cases

ultimate.loc[ultimate["Intrnl"] > 0, :] *= 2 # Multiply the time taken by 2 for any bank which investigates alone when bank_to is International

full = full.join(ultimate) # Combine with full dataset

####################################################################################################################

temp2 = np.zeros((np.shape(full2)[0], 6))
temp2 = pd.DataFrame(temp2, columns = ['bank_A', 'bank_B', 'bank_C', 'bank_D', 'bank_E', 'Intrnl']) 
bank_to_df = temp2.reset_index()
bank_to_df.update(full2.pivot(columns = 'bank_to', values = 'time_spent')) # Pivot to get the time spent for each bank_to
bank_to_df = bank_to_df.set_index('index').rename_axis(None)
bank_to_df
bank_from_df = temp2.reset_index()
bank_from_df.update(full2.pivot(columns = 'bank_from', values = 'time_spent')) # Pivot to get the time spent for each bank_from
bank_from_df = bank_from_df.set_index('index').rename_axis(None)
ultimate2 = bank_to_df/2 + bank_from_df/2 # Halve and add them: this forces the 50:50 split for sharing cases
ultimate2.loc[ultimate2["Intrnl"] > 0, :] *= 2 # Multiply the time taken by 2 for any bank which investigates alone when bank_to is International
full2 = full2.join(ultimate2) # Combine with full dataset

####################################################################################################################

# Data Initialisation
teamsize = [8, 12, 10, 10, 10]
bank_names = ['bank_A', 'bank_B', 'bank_C', 'bank_D', 'bank_E']

time = [0.25, 0.5, 1, 2]
ext_cost = [40, 60, 100, 150]

n_priority = len(time)
Priorities = range(n_priority)

# Index Sets
n_banks = 5
Banks = range(n_banks)

Categories = range(np.shape(one_hot)[1])

## Combined mega dataframe ##
training = pd.concat([full, full2], ignore_index = True)
training["day"] = (training["date"] - training["date"][0]).dt.days

training["start_month"] = 0
training["mid_month"] = 0
training["end_month"] = 0
training.loc[training.date.dt.day <= 10, "start_month"] = 1
training.loc[(training.date.dt.day > 10) & (training.date.dt.day <= 20), "mid_month"] = 1
training.loc[(training.date.dt.day > 20), "end_month"] = 1

# Begin loop
n_training_days = len(np.unique(full["day"]))

testing_days_used = np.unique(full2["day"])[0:2] # change this number to change the no. of testing days
n_testing_days = len(testing_days_used)


random.seed(4)
loss_value_mem = []
n_scam_undetected = []

gain_value_mem = []
n_scam_detected = []

for q in range(n_testing_days):
    
    test_day = n_training_days + q

    print("*************************************************************************")
    print("Solving model for test day", test_day, "...")
    

    # full is testing dataset
    full = training.loc[training["day"] < test_day].reset_index(drop = True)

    # full2 is testing dataset
    full2 = training.loc[training["day"] == test_day].reset_index(drop = True)

    # Index sets
    n_days = len(pd.unique(full["date"]))
    Days = range(n_days)
    n_daily_cases = np.shape(full)[0]
    Cases = range(n_daily_cases)

    # Problem set-up
    prob_full = xp.problem('prob_full')
    prob_full.setControl('outputlog', 0) # suppress output

    weight_f = np.array([xp.var(vartype = xp.continuous, name = 'weight_{0}'.format(i+1), lb = -xp.infinity) 
                        for i in Categories], dtype = xp.npvar) # now with negative weights possible. Later defined to be <= 0.9
    transact_w_f = xp.var(vartype = xp.continuous, name='transact_w_f') # 0 to 1
    cust_w_f = xp.var(vartype = xp.continuous, name='cust_w_f') # 0 to 1
    amount_w_f = xp.var(vartype = xp.continuous, name='amount_w_f')
    start_month_w = xp.var(vartype = xp.continuous, name = 'start_month_w')
    mid_month_w = xp.var(vartype = xp.continuous, name = 'mid_month_w')
    end_month_w = xp.var(vartype = xp.continuous, name = 'end_month_w')
    combined = np.array([xp.var(vartype = xp.continuous, name = 'combined_{0}'.format(i+1))
                for i in Cases], dtype = xp.npvar) # 0 to 1

    internal = np.array([xp.var(vartype = xp.continuous, name = 'internal_{0}_{1}_{2}'.format(i+1, j+1, k+1))
                for i in Days for j in Banks for k in Priorities], dtype = xp.npvar).reshape(n_days, n_banks, n_priority) # Expected no. of days of internal investigatins
    external = np.array([xp.var(vartype = xp.continuous, name = 'external_{0}_{1}_{2}'.format(i+1, j+1, k+1))
                for i in Days for j in Banks for k in Priorities], dtype = xp.npvar).reshape(n_days, n_banks, n_priority) # Expected no. of external investigators of that priority.

    prob_full.addVariable(weight_f, cust_w_f, transact_w_f, amount_w_f, start_month_w, mid_month_w, end_month_w, combined, internal, external) 

    # Constraints

    # Combined weights
    combined_cons = [combined[i] == xp.Sum(weight_f[c] * full.iloc[i, 17+c] for c in Categories) + 
                     transact_w_f * full["transac_prob"][i] + cust_w_f * full["customer_prob"][i] + 
                     amount_w_f * full["Amount"][i] + 
                     start_month_w * full["start_month"][i] + mid_month_w * full["mid_month"][i] + 
                     end_month_w * full["end_month"][i] for i in Cases]
    probab_invst_cons = [combined[i] <= 1 for i in Cases] # 0 <= Combined weights <= 1
    prob_full.addConstraint(combined_cons, probab_invst_cons) 
    for i in Cases:
        for b in Banks:
            full[bank_names[b]][i] = full[bank_names[b]][i] * combined[i] # Multiplying decision variable with time spent.
    
    # Find sum by day by bank by priority
    temptemptemp = full.groupby(["priority", "day"])
    A_df = pd.DataFrame({'bank_A': temptemptemp["bank_A"].sum()}).reset_index()
    B_df = pd.DataFrame({'bank_B': temptemptemp["bank_B"].sum()}).reset_index()
    C_df = pd.DataFrame({'bank_C': temptemptemp["bank_C"].sum()}).reset_index()
    D_df = pd.DataFrame({'bank_D': temptemptemp["bank_D"].sum()}).reset_index()
    E_df = pd.DataFrame({'bank_E': temptemptemp["bank_E"].sum()}).reset_index()

    A_df["bank_B"] = B_df["bank_B"]
    A_df["bank_C"] = C_df["bank_C"]
    A_df["bank_D"] = D_df["bank_D"]
    A_df["bank_E"] = E_df["bank_E"]
    exp_work_df = A_df
    
    for d in Days:
        for b in Banks:
            for p in Priorities:
                prob_full.addConstraint(exp_work_df.iloc[(p-1+1)*(n_days) + d, b+2] <= internal[d,b,p] + (external[d,b,p] * time[p])) # Time constraint for expected investigation

    for d in Days:
        for b in Banks:
            prob_full.addConstraint(xp.Sum(internal[d,b,p] for p in Priorities) == teamsize[b]) # All internal investigation days for each bank mus be less than their team size.

    for c in Categories:
        prob_full.addConstraint(weight_f[c] <= 0.9) # Upper bound on weights as 0.9

    # Objective and solve
    prob_full.setObjective(xp.Sum(external[d,b,p] * ext_cost[p] for d in Days for b in Banks for p in Priorities if d < (test_day - 30)) + 
                           xp.Sum((1-combined[i]) * full["is_scam"][i] * full["Amount"][i] for i in Cases if full["day"][i] < (test_day - 30)) +
                           5 * xp.Sum(external[d,b,p] * ext_cost[p] for d in Days for b in Banks for p in Priorities if d >= (test_day - 30)) + 
                           5 * xp.Sum((1-combined[i]) * full["is_scam"][i] * full["Amount"][i] for i in Cases if full["day"][i] >= (test_day - 30)), sense = xp.minimize)
    prob_full.solve()
    print(f'The objective function value for day {test_day} is {prob_full.getObjVal()}')

    transact_sol = prob_full.getSolution(transact_w_f)
    cust_sol = prob_full.getSolution(cust_w_f)
    weight_sol = prob_full.getSolution(weight_f)
    amount_sol = prob_full.getSolution(amount_w_f)
    start_month_sol = prob_full.getSolution(start_month_w)
    mid_month_sol = prob_full.getSolution(mid_month_w)
    end_month_sol = prob_full.getSolution(end_month_w)
    print("The weights on the categories are:", weight_sol)
    print("The weights on the transaction probabilities are:", transact_sol)
    print("The weights on the customer probabilities are:", cust_sol)
    print("The weights on the transaction value are:", amount_sol)
    print("The weights on start, mid, and end month are:", start_month_sol, mid_month_sol, end_month_sol)

    # print(prob_full.getSolution(combined))
    # print(prob_full.getSolution(internal))
    # print(prob_full.getSolution(external))

    
    vec = [amount_sol, transact_sol, cust_sol]
    vec_mult_frame = full2.iloc[:,[2,11,15]] * vec
    weight_sol_proc_mult_frame = full2.iloc[:,17:37] * weight_sol
    month_vec = [start_month_sol, mid_month_sol, end_month_sol]
    month_vec_mult_frame = full2.iloc[:, -3:] * month_vec
    joined_mult_frame = vec_mult_frame.join(weight_sol_proc_mult_frame)
    joined_mult_frame = joined_mult_frame.join(month_vec_mult_frame)

    combined_day2 = joined_mult_frame.sum(axis = 1)
    combined_day2.loc[combined_day2 > 1] = 1
    combined_day2.loc[combined_day2 < 0] = 0
    # sample from combined_day2

    sample = np.random.binomial(1,combined_day2)
    to_investigate = [i for i in range(len(sample)) if sample[i]==1]
    full2['to_investigate'] = sample
    # full2.loc[full2["to_investigate"] == 1] # Uncomment to see all cases investigated

    #Scams detected:
    # print(full2[(full2["to_investigate"] == 1) & (full2["is_scam"] == 1)]) # Uncomment to find scams detected
    
    # Find sum by bank by priority
    temptemptemp2 = full2.loc[full2["to_investigate"]==1].groupby(["priority", "day"])
    A_df = pd.DataFrame({'bank_A': temptemptemp2["bank_A"].sum()}).reset_index()
    B_df = pd.DataFrame({'bank_B': temptemptemp2["bank_B"].sum()}).reset_index()
    C_df = pd.DataFrame({'bank_C': temptemptemp2["bank_C"].sum()}).reset_index()
    D_df = pd.DataFrame({'bank_D': temptemptemp2["bank_D"].sum()}).reset_index()
    E_df = pd.DataFrame({'bank_E': temptemptemp2["bank_E"].sum()}).reset_index()

    A_df["bank_B"] = B_df["bank_B"]
    A_df["bank_C"] = C_df["bank_C"]
    A_df["bank_D"] = D_df["bank_D"]
    A_df["bank_E"] = E_df["bank_E"]
    exp_work_df2 = A_df
    # print(exp_work_df2)
    subtract_exp_work_df2 = exp_work_df2

    # Subtract from low priority first
    # WARNING this code assumes that there's only one day being investigated
    for b in Banks:
        remaining = teamsize[b]
        for p in range(np.shape(subtract_exp_work_df2)[0]): # Loop over priorities; this is necessary because sometimes the model doesnt investigate prio 4
            if (remaining > 0):
                if (subtract_exp_work_df2.iloc[p,2+b] - remaining < 0):
                    remaining = remaining - subtract_exp_work_df2.iloc[p,2+b]
                    subtract_exp_work_df2.iloc[p,2+b] = 0
                else: 
                    subtract_exp_work_df2.iloc[p,2+b] = subtract_exp_work_df2.iloc[p,2+b] - remaining
                    remaining = 0
    # print(subtract_exp_work_df2)
    spending = 0
    for p in range(np.shape(subtract_exp_work_df2)[0]): # Loop over priorities; this is necessary because sometimes the model doesnt investigate prio 4
        spending += sum(np.ceil(subtract_exp_work_df2.iloc[p, 2:7] / time[p])) * ext_cost[p]
    spending

    # Amount gained from detecting scams
    print("Gain from the detected scams is:", sum(full2[(full2["to_investigate"] == 1) & (full2["is_scam"] == 1)]["Amount"]))

    # Actual loss from undetected scams
    print("Loss from scams undetected is:", sum(full2[(full2["to_investigate"] == 0) & (full2["is_scam"] == 1)]["Amount"]))

    # Total loss after accounting for external investigators
    print("Total loss + spending:",  sum(full2[(full2["to_investigate"] == 0) & (full2["is_scam"] == 1)]["Amount"]) + spending)

    altspending = 0
    for p in range(np.shape(subtract_exp_work_df2)[0]): # Loop over priorities; this is necessary because sometimes the model doesnt investigate prio 4
        altspending += np.ceil(sum(subtract_exp_work_df2.iloc[p, 2:7] / time[p])) * ext_cost[p]

    n_priority_1_invst = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 1) & (full2["priority"] == 1)])[0]
    n_priority_2_invst = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 1) & (full2["priority"] == 2)])[0]
    n_priority_3_invst = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 1) & (full2["priority"] == 3)])[0]
    n_priority_4_invst = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 1) & (full2["priority"] == 4)])[0]

    n_priority_1_scam = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 0) & (full2["priority"] == 1)])[0]
    n_priority_2_scam = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 0) & (full2["priority"] == 2)])[0]
    n_priority_3_scam = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 0) & (full2["priority"] == 3)])[0]
    n_priority_4_scam = np.shape(full2.loc[(full2["is_scam"] == 1) & (full2["to_investigate"] == 0) & (full2["priority"] == 4)])[0]
    # Better loss if we better allocate external investigators
    print("Better total loss + spending by better allocating external investigators:",  sum(full2[(full2["to_investigate"] == 0) & (full2["is_scam"] == 1)]["Amount"]) + altspending)

    end = tm.time()
    loss_value_mem.append(sum(full2[(full2["to_investigate"] == 0) & (full2["is_scam"] == 1)]["Amount"]))
    n_scam_undetected.append([n_priority_1_scam, n_priority_2_scam, n_priority_3_scam, n_priority_4_scam])
    gain_value_mem.append(sum(full2[(full2["to_investigate"] == 1) & (full2["is_scam"] == 1)]["Amount"]) + altspending)
    n_scam_detected.append([n_priority_1_invst, n_priority_2_invst, n_priority_3_invst, n_priority_4_invst])

    print(f"Time since start of code: {end - start:0.4f} seconds")
n_scam_detected = np.array(n_scam_detected)
n_scam_undetected = np.array(n_scam_undetected)
loss_value_mem = np.array(loss_value_mem)
gain_value_mem = np.array(gain_value_mem)
prio1_undet = n_scam_undetected[:, 0]
prio2_undet = n_scam_undetected[:, 1]
prio3_undet = n_scam_undetected[:, 2]
prio4_undet = n_scam_undetected[:, 3]
prio1_det = n_scam_detected[:, 0]
prio2_det = n_scam_detected[:, 1]
prio3_det = n_scam_detected[:, 2]
prio4_det = n_scam_detected[:, 3]

# p.savetxt('0-9days.csv', [p for p in zip(testing_days_used, gain_value_mem, loss_value_mem, prio1_det, prio2_det, prio3_det, prio4_det,
#                                          prio1_undet, prio2_undet, prio3_undet, prio4_undet)], delimiter=',', fmt='%s',
#                                          header="test_day,gain_value,loss_value,prio1_det,prio2_det,prio3_det,prio4_det,prio1_undet,prio2_undet,prio3_undet,prio4_undet", 
#                                          comments="")