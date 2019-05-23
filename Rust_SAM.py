import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as mpl
from scipy.sparse import diags


def utility_engine(theta, i, x_t): #theta[2] is replacement engine cost
    return i * (-1) * theta[2] + (1 - i) * (-1)*(theta[0] * x_t + theta[1] * x_t ** 2)


def prob_replace(EV, theta, beta):
    x_t = np.linspace(0, 10, 11).astype(int) # initialize bus, START 'ER UP
    EV = EV.reshape(1, 11)

    V1 = np.exp(utility_engine(theta, i=1, x_t=x_t) + beta * EV[0, 0])
    V0 = np.exp(utility_engine(theta, i=0, x_t=x_t) + beta * EV[0, x_t])

    return V1 / (V1 + V0)


# Choice-specific value function calculation
# log( sum( exp( ) ) ) is for taking the expected maximum of a T1EV random value plus the deterministic utility
def choice_specific(epsilon, max_iter, beta, theta):
    x_t = np.linspace(0, 10, 11)
    dif = 1e5
    iterNum = 0
    P = [0.2, 0.8]
    P = diags(P, np.arange(2), shape=(11, 11)).todense()
    P = P.T
    P[10, 10] = 1
    EV = np.zeros(11)
    mu = 0.5772
    while iterNum < max_iter and dif > epsilon:
        EV = EV.reshape(1, 11)

        if np.max(utility_engine(theta=theta, i=1, x_t=x_t) + beta * EV[0,0]) > 500: # For weird theta draws
            break

        new_value = (mu + np.log(np.exp(utility_engine(theta=theta, i=0, x_t=x_t) + beta * EV) +
                          np.exp(utility_engine(theta=theta, i=1, x_t=x_t) + beta * EV[0,0]))) * P


        iterNum += 1
        dif = np.linalg.norm(new_value - EV)
        EV = new_value


    return EV

EV = choice_specific(1e-10, 300, 0.95, [0.3, 0.0, 4.0])

# Calculate replacement probability

prob_rep = prob_replace(EV=EV, theta=[0.3, 0.0, 4.0], beta=0.95)
# print(prob_rep)


# Make uniform draws and transform them to T1EV
# Simulate data to do MLE inference on structural parameters

def sim_data(T, theta, EV, beta):
    draws = np.random.uniform(0, 1, T*2).reshape(T, 2)
    draws = -np.log(-np.log(draws))
    x_t = np.linspace(0, 10, 11)
    util_dont_replace = utility_engine(theta=theta, i=0, x_t=x_t)
    util_replace = utility_engine(theta=theta, i=1, x_t=x_t)
    rep_mat = np.zeros((T + 1, 2))

    x_t = np.zeros(T + 1)

    for i in range(1, T):
        x_t[i] = x_t[i - 1]
        if x_t[i] > 10:
            x_t[i] = 10
        index = int(x_t[i])
        mileage_jump = np.random.binomial(1, 0.8, 1)

        V0 = util_dont_replace[index] + draws[i, 0] + beta * EV[0, index]
        V1 = util_replace[index] + draws[i, 1] + beta * EV[0, 0]

        decision = V0 - V1

        if decision > 0:
            x_t[i] = x_t[i] + mileage_jump

        else:
            x_t[i] = mileage_jump

        rep_mat[i, 0] = (decision < 0) # =1 if replace
        rep_mat[i+1, 1] = x_t[i]
        if rep_mat[i+1, 1] > 10:
            rep_mat[i+1, 1] = 10

    return rep_mat

# Make sure you set T high enough so every mileage bin gets draws, it takes a while for LLN to do some work
rep_mat = sim_data(T=1000000, theta=[0.3, 0.0, 4.0], EV=EV, beta=0.95)

# Get "empirical" replacement probabilities

def emp_rep(sim_mat):
    emp_rep_mat = np.zeros((11,2)) # first column number of replacements, second the number of times in that bin
    l = sim_mat.shape[0]
    for i in range(l):
        for j in range(11):
            if sim_mat[i, 1] == j:
                emp_rep_mat[j, 0] += sim_mat[i,0] # if replace = 1
                emp_rep_mat[j, 1] += 1
                break

    emp_rep_prob = emp_rep_mat[:, 0] / emp_rep_mat[:, 1]

    return emp_rep_prob, emp_rep_mat


est_rep_prob = emp_rep(rep_mat)[0]

grid = np.linspace(0, 10, 11)

# Ideally these two should converge completely, need around T=1M for the right tail to converge well
mpl.plot(grid, prob_rep.T, label="analytical CS")
mpl.title("Probability of replacing at a given mileage")
mpl.xlabel("Mileage bin")
mpl.ylabel("Density")
mpl.plot(grid, est_rep_prob, label="simulated")
mpl.legend()
mpl.show()

def log_lik(theta, EV, beta, lam, mileage, replacement):
    ll = 0
    for i in range(1, 5000):
        delta_mileage = mileage[i] - mileage[i - 1]
        if delta_mileage == 1 and replacement[i - 1] == 0:
            prob_mileage = lam
        elif delta_mileage == 0 and replacement[i - 1] == 0:
            prob_mileage = 1 - lam
        elif delta_mileage == 1 and replacement[i - 1] == 1:
            prob_mileage = lam
        elif mileage[i] == 0:
            prob_mileage = 1 - lam
        elif mileage[i] == 1 and replacement[i - 1] == 1:
            prob_mileage = lam

        else:
            print("Weird mileage transition occured")
            print(mileage[i], delta_mileage, replacement[i])

        x_t = int(mileage[i])
        prob_rep = prob_replace(EV=EV, theta=theta, beta=beta)
        prob_rep_xt = prob_rep[0, x_t]

        if replacement[i] == 1:
            prob_rep_engine = prob_rep_xt
        elif replacement[i] == 0:
            prob_rep_engine = 1 - prob_rep_xt
        else:
            print("Error in rep_prob")

        if prob_rep_engine <= 0: # For bad theta draws you need an exception so the log function doesn't break
            prob_rep_engine = 0.0001
            print("Warning, prob_rep_engine set incorrectly")

        ll_temp = np.log(prob_rep_engine) + np.log(prob_mileage)
        ll += ll_temp
    return -ll


# Simulate new data for MLE procedure
s_data = sim_data(5000, theta=[0.3, 0.0, 4.0], EV=EV, beta=0.95)
replacement = s_data[:, 0]
mileage = s_data[:, 1]
def run_rust(theta, beta, lam, replacement, mileage):
    # Get EVs
    EV = choice_specific(epsilon=1e-10, max_iter=1000, beta=beta, theta=theta)

    # Get log-likelihood
    ll = log_lik(theta=theta, EV=EV, beta=beta, lam=lam, mileage=mileage, replacement=replacement)
    print("One Loop")
    print(theta)
    print(ll)

    return ll


theta_init = np.array([0.3, 0.0, 4.0])

MLE = minimize(run_rust, theta_init, method='BFGS', args=(0.95, 0.8, replacement, mileage)) #BFGS seems faster
print(MLE)
# Sometimes it says success: false, but the precision is extremely good anyway