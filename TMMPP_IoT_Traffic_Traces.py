import numpy as np
import random
import matplotlib.pyplot as plt
###############################################
res = 5 #resolution in min
seed = 98
###############################################
#BETA PDF: 
BETA = 0.0167 #Beta function constant for (3,4)
def BETA_Pdf(T,t):
    alpha = 3
    beta = 4
    beta_pdf = (1/BETA)* ((t**(alpha-1))*((T-t)**(beta-1))/(T**(alpha+beta-1)))
    return(beta_pdf)
##################################################
#Normal Distribution:
def Norm_Pdf(dn,mu):
    sigma = 1
    Norm_pdf = (1/(sigma*np.sqrt(2*3.14)))* (np.exp(-0.5*(((dn-mu)/sigma)**2)))
    return(Norm_pdf)
##################################################
#Exponential Distribution:
def Exp_Pdf(t):
    l = 0.004
    Exp_pdf = 0.9*np.exp(-l*t)
    return(Exp_pdf)
####################################################
#TIER 1 PARAMETERS:
node_density = int(1000) #number of IoT devices in one cell
########################################################
n_node_states = 2
node_state_range = np.arange(0,n_node_states)
node_state_range = np.reshape(node_state_range,(n_node_states))
Pu = np.array([[1,0],[1,0]])
Pc = np.array([[0,1],[1,0]])
lambda_sn1 = 0.3
lambda_sn2 = 10*lambda_sn1
lambda_states =[lambda_sn1,lambda_sn2]
########################################################
#TIER 2 PARAMETERS:
n_System_states = 3
System_state_range = np.arange(0,n_System_states)
System_state_range = np.reshape(System_state_range,(n_System_states))
C_low = np.array([[1,0,0],[1,0,0],[1,0,0]])
C_high = np.array([[0,1,0],[0,1,0],[0,1,0]])
p11= 0.9
p12 = 0.1
C_D = np.array([[0,1,0],[0,p11,p12],[0,1,0]])
##
hour = 60 #min
day = 24*hour
Ts= int(24*hour/res) #seosanility period 
Te = int((16*hour)/res)
TD = Ts - Te
Ta = int(1*hour/res)
Tb = int(1*hour/res)
M = int(TD/Ta)
State_period_range = np.array([Te,Ta,Tb])
P_s0= 0.4
P_s1= 0.5
P_s2 = 0.6
##
lambda_s1 = 0.8
#############################################################
## Time Series:
K = 5##number of days in the time-series duration 
Time_Series_Duration = int((K*day)/res) #number of min in the time-series duration
##
ti= 0
t = 0
n = 0 
k = 0
m = 0
delta1 = 0
delta2 = 0
delta3 = 0
##
#Q = np.zeros(Time_Series_Duration)
System_states = [0]
Node_states = np.zeros((Time_Series_Duration,node_density))
##
Traffic_nodes = 0 
Traffic_n = np.zeros((node_density,Time_Series_Duration))
Traffic_volume = np.zeros(Time_Series_Duration)
Agregated_packets = np.zeros(Time_Series_Duration)
theta_Ts= np.zeros((Time_Series_Duration,node_density))
##
theta_t_s2=np.zeros((Tb))
###########
while ti in range(Time_Series_Duration):
    #Calculate P_G(t)
    k=0
    delta1=0
    delta2=0
    delta3=0
    for k in range(K):
        if (ti-(k*24*hour/res))==0:
            delta1 = 1
            break
        elif (ti-k*Ts-Te)==0:
            delta2 = 1
            break
        else:
            for m in range (M):
                if (ti-k*Ts-Te-(m*Ta))==0 :##Assuming Ta=Tb
                    delta3 = 1
                    break 
                m+=1
        k+=1
    Q_t = delta1*C_low+delta2*C_high+delta3*C_D
    #Update the System State:
    Sys_si = System_states[-1] #initial system state
    #update the state:
    Q_weights = Q_t[int(Sys_si),:]
    random.seed(seed+ti)     
    Sys_sn = random.choices(System_state_range, Q_weights,k=1)
    Sys_sn = Sys_sn[0]
    if Sys_sn==2:
        tf=0
    ##
    System_states.append(Sys_sn)
    ##
    State_period = State_period_range[Sys_sn]
    ##
    t=0
    for t in range(State_period):
        ##Theta(s,t):
        if Sys_sn==2:
           theta_t_s2[t] = BETA_Pdf(Tb,t)
           lambda_s2 = lambda_s1+theta_t_s2[t] 
           random.seed(t+seed)
           Agregated_packets_n = np.random.poisson(lambda_s2,int(node_density*P_s2)) 
           Agregated_packets[t+ti] = np.sum(Agregated_packets_n)
        elif Sys_sn==1:
             random.seed(t+seed)
             Agregated_packets_n = np.random.poisson(lambda_s1,int(node_density*P_s1))
             Agregated_packets[t+ti] = np.sum(Agregated_packets_n)
             Aggregated_alert = Agregated_packets[t+ti] 
        elif Sys_sn==0:
             random.seed(t+ti+seed)
             theta_t = Exp_Pdf(t)
             lambda_s0 = theta_t
             random.seed(t+ti+seed)
             Agregated_packets_n = np.random.poisson(lambda_s0,int(node_density*P_s0))
             Agregated_packets[t+ti] = np.sum(Agregated_packets_n)
        np.random.seed(t+seed)
        Packet_sizes = (np.random.pareto(3.4, int(Agregated_packets[t+ti])) + 3.5) * 5
        Traffic_volume[ti+t] = sum(Packet_sizes)
        t+=res
    ti+=State_period

#############################################################
#Plot Normalized Traffic:
plt.plot(Agregated_packets/max(Agregated_packets))
#############################################################
#To save a MATLAB File of Traffic Traces:
# from scipy.io import savemat
# mymat1={'Traffic':np.array(Agregated_packets)}
# savemat("Traffic.mat", mymat1)
