# from collections import defaultdict
import numpy as np
from copy import deepcopy
from PCMCI_urban.CMI_estimator import CMI_KNN
from pcmciminus import pcmcim 
# from scipy.stats import pearsonr, spearmanr
from itertools import combinations
# from scipy.integrate import odeint
import scipy.io
# from data_processing import DataFrame


xi = scipy.io.loadmat('/Users/tg2426/Documents/Python/Covid/Synthetic_Results/case14.mat')
A = xi['A']

XI_X = xi['XI_X']
XI_Y = xi['XI_Y']

ZETA_X = xi['ZETA_X']
ZETA_Y = xi['ZETA_Y']

BETA_X = xi['BETA_X']
BETA_Y = xi['BETA_Y']

TE_XY = xi['TE_XY']
TE_YX = xi['TE_YX']


B_X = xi['B_X']
B_Y = xi['B_Y']

M1,M2,N,T = XI_X.shape

print(xi['gamma_x'],xi['gamma_y'], N, T, A)
print(TE_XY, TE_YX)

TE_XY_XI = np.zeros((M1,M2),dtype=float)
TE_YX_XI = np.zeros((M1,M2),dtype=float)

PVALUE_XY = np.zeros((M1,M2),dtype=float)
PVALUE_YX = np.zeros((M1,M2),dtype=float)


# array of FAMIs
# N = 1
array_zeta = np.zeros((N,2,T),dtype=float)

array_zeta[:,0,:] = ZETA_X
array_zeta[:,1,:] = ZETA_Y

# array of SAMIs
array_xi = np.zeros((N,2,T),dtype=float)
array_xi[:,0,:] = XI_X[0,1]
array_xi[:,1,:] = XI_Y[0,1]





tau_min = 1                          
tau_max = 3

# # print(array_zeta)
# print("SAMIs")

# FAMIs_covid = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/FAMIs_covid_weekly.npy')
# FAMIs_death = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/FAMIs_death_weekly.npy')
# FAMIs_vaccine = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/FAMIs_vaccine_weekly.npy')
# # FAMIs_vaccine_dose1 = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/FAMIs_vaccine_dose1.npy')

# # SAMIs_covid = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/SAMIs_covid.npy')
# # SAMIs_death = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/SAMIs_death.npy')
# # SAMIs_vaccine = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/SAMIs_vaccine.npy')
# # SAMIs_vaccine_dose1 = np.load('/Users/tg2426/Documents/Python/Covid/PCMCI_urban/SAMIs_vaccine_dose1.npy')

# N,T = FAMIs_covid.shape
# print(N,T)

# # array of FAMIs
# # N = 1
# array_FAMI = np.zeros((N,3,T),dtype=float)

# array_FAMI[:,0,:] = FAMIs_covid
# array_FAMI[:,1,:] = FAMIs_death
# array_FAMI[:,2,:] = FAMIs_vaccine
# # array_FAMI[:,3,:] = FAMIs_vaccine_dose1

# # array of SAMIs
# array_SAMI = np.zeros((N,4,T),dtype=float)

# # array_SAMI[:,0,:] = SAMIs_covid
# # array_SAMI[:,1,:] = SAMIs_death
# # array_SAMI[:,2,:] = SAMIs_vaccine
# # array_SAMI[:,3,:] = SAMIs_vaccine_dose1

# N,dim,T = array_FAMI.shape
# # r = 0
# # # constraint_dim = constraint.shape[1]
# # # while r<dim-1:
# # for i in range(dim):
# #     for j in range(dim):
# # # for target in list(combinations(range(dim),2)):
# #         # i, j = target
# #         parent = (i, -1)
# #         Z = [(j, -1)]
# #         if i!=j:
# #             cmi = CMI_KNN(array=array_FAMI, sig_samples=5, shuffle_neighbors=20, knn=5)                    
# #             val = cmi.independence_measure(array = array_FAMI, X = [parent], Y = [(j, 0)], Z = Z)
# #             pval = cmi.get_shuffle_significance(X = [parent], Y = [(j, 0)], Z = Z, value= val)
# #             print(i,j,Z, val, pval)


pcmcim(array_zeta, sig_samples=1000, shuffle_neighbors=5, knn=9).run_pcmci(tau_max=tau_max,max_combinations=1,  pc_alpha=0.05)
# # # pcmcim(array_zeta).get_lagged_dependencies(tau_min=tau_min,tau_max=tau_max)
# # # print(array_xi.shape)
# # print("SAMIs")
# # pcmcim(array_xi).get_lagged_dependencies(tau_min=tau_min,tau_max=tau_max)




# # def _reverse_link(self, link):
# #         """Reverse a given link, taking care to replace > with < and vice versa."""

# #         if link == "":
# #             return ""

# #         if link[2] == ">":
# #             left_mark = "<"
# #         else:
# #             left_mark = link[2]

# #         if link[0] == "<":
# #             right_mark = ">"
# #         else:
# #             right_mark = link[0]

# #         return left_mark + link[1] + right_mark
# # def _get_int_parents(parents):
# #     """Get the input parents dictionary.

# #     Parameters
# #     ----------
# #     parents : dict or None
# #         Dictionary of form {0:[(0, -1), (3, -2), ...], 1:[], ...}
# #         specifying the conditions for each variable. If None is
# #         passed, no conditions are used.

# #     Returns
# #     -------
# #     int_parents : defaultdict of lists
# #         Internal copy of parents, respecting default options
# #     """
# #     int_parents = deepcopy(parents)
# #     if int_parents is None:
# #         int_parents = defaultdict(list)
# #     else:
# #         int_parents = defaultdict(list, int_parents)
# #     return int_parents


# # def _iter_indep_conds(_int_parents, _int_link_assumptions, max_conds_py,max_conds_px):
# #     for j in range(N):
# #     # Get the conditions for node j
# #         conds_y = _int_parents[j][:max_conds_py]
# #         # Create a parent list from links seperated in time and by node
# #         # parent_list = [(i, tau) for i, tau in _int_link_assumptions[j]
# #         #                if (i, tau) != (j, 0)]
# #         parent_list = []
# #         for itau in _int_link_assumptions[j]:
# #             link_type = _int_link_assumptions[j][itau]
# #             if itau != (j, 0) and link_type not in ['<--', '<?-']:
# #                 parent_list.append(itau)
# #         for cnt, (i, tau) in enumerate(parent_list):
# #             # Get the conditions for node i
# #             conds_x = _int_parents[i][:max_conds_px]
# #             # Shift the conditions for X by tau
# #             conds_x_lagged = [(k, tau + k_tau) for k, k_tau in conds_x]
# #             # Print information about the mci conditions if requested
# #             # if verbosity > 1:
# #             # _print_mci_conditions(conds_y, conds_x_lagged, j, i,
# #             #                             tau, cnt, len(parent_list))
# #             # Construct lists of tuples for estimating
# #             # I(X_t-tau; Y_t | Z^Y_t, Z^X_t-tau)
# #             # with conditions for X shifted by tau
# #             Z = [node for node in conds_y if node != (i, tau)]
# #             # Remove overlapped nodes between conds_x_lagged and conds_y
# #             Z += [node for node in conds_x_lagged if node not in Z]
# #             # Yield these list
# #             # yield j, i, tau, Z
# #             # print(conds_y)
# #             yield j,i, tau, Z



# # def convert_to_string_graph(graph_bool):
# #     """Converts the 0,1-based graph returned by PCMCI to a string array
# #     with links '-->'.

# #     Parameters
# #     ----------
# #     graph_bool : array
# #         0,1-based graph array output by PCMCI.

# #     Returns
# #     -------
# #     graph : array
# #         graph as string array with links '-->'.
# #     """

# #     graph = np.zeros(graph_bool.shape, dtype='<U3')
# #     graph[:] = ""
# #     # Lagged links
# #     graph[:,:,1:][graph_bool[:,:,1:]==1] = "-->"
# #     # Unoriented contemporaneous links
# #     graph[:,:,0][np.logical_and(graph_bool[:,:,0]==1, 
# #                                 graph_bool[:,:,0].T==1)] = "o-o"
# #     # Conflicting contemporaneous links
# #     graph[:,:,0][np.logical_and(graph_bool[:,:,0]==2, 
# #                                 graph_bool[:,:,0].T==2)] = "x-x"
# #     # Directed contemporaneous links
# #     for (i,j) in zip(*np.where(
# #         np.logical_and(graph_bool[:,:,0]==1, graph_bool[:,:,0].T==0))):
# #         graph[i,j,0] = "-->"
# #         graph[j,i,0] = "<--"

# #     return graph


# # def symmetrize_p_and_val_matrix( p_matrix, val_matrix, link_assumptions, conf_matrix=None):
# #         """Symmetrizes the p_matrix, val_matrix, and conf_matrix based on link_assumptions
# #            and the larger p-value.

# #         Parameters
# #         ----------
# #         val_matrix : array of shape [N, N, tau_max+1]
# #             Estimated matrix of test statistic values.
# #         p_matrix : array of shape [N, N, tau_max+1]
# #             Estimated matrix of p-values. Set to 1 if val_only=True.
# #         conf_matrix : array of shape [N, N, tau_max+1,2]
# #             Estimated matrix of confidence intervals of test statistic values.
# #             Only computed if set in cond_ind_test, where also the percentiles
# #             are set.
# #         link_assumptions : dict or None
# #             Dictionary of form {j:{(i, -tau): link_type, ...}, ...} specifying
# #             assumptions about links. This initializes the graph with entries
# #             graph[i,j,tau] = link_type. For example, graph[i,j,0] = '-->'
# #             implies that a directed link from i to j at lag 0 must exist.
# #             Valid link types are 'o-o', '-->', '<--'. In addition, the middle
# #             mark can be '?' instead of '-'. Then '-?>' implies that this link
# #             may not exist, but if it exists, its orientation is '-->'. Link
# #             assumptions need to be consistent, i.e., graph[i,j,0] = '-->'
# #             requires graph[j,i,0] = '<--' and acyclicity must hold. If a link
# #             does not appear in the dictionary, it is assumed absent. That is,
# #             if link_assumptions is not None, then all links have to be specified
# #             or the links are assumed absent.
# #         Returns
# #         -------
# #         val_matrix : array of shape [N, N, tau_max+1]
# #             Estimated matrix of test statistic values.
# #         p_matrix : array of shape [N, N, tau_max+1]
# #             Estimated matrix of p-values. Set to 1 if val_only=True.
# #         conf_matrix : array of shape [N, N, tau_max+1,2]
# #             Estimated matrix of confidence intervals of test statistic values.
# #             Only computed if set in cond_ind_test, where also the percentiles
# #             are set.
# #         """

# #         # Symmetrize p_matrix and val_matrix and conf_matrix
# #         for i in range(N):
# #             for j in range(N):
# #                 # If both the links are present in link_assumptions, symmetrize using maximum p-value
# #                 # if ((i, 0) in link_assumptions[j] and (j, 0) in link_assumptions[i]):
# #                 if (i, 0) in link_assumptions[j]:
# #                     if link_assumptions[j][(i, 0)] in ["o-o", 'o?o']:
# #                         if (p_matrix[i, j, 0]
# #                                 >= p_matrix[j, i, 0]):
# #                             p_matrix[j, i, 0] = p_matrix[i, j, 0]
# #                             val_matrix[j, i, 0] = val_matrix[i, j, 0]
# #                             if conf_matrix is not None:
# #                                 conf_matrix[j, i, 0] = conf_matrix[i, j, 0]

# #                     # If only one of the links is present in link_assumptions, symmetrize using the p-value of the link present
# #                     # elif ((i, 0) in link_assumptions[j] and (j, 0) not in link_assumptions[i]):
# #                     elif link_assumptions[j][(i, 0)] in ["-->", '-?>']:
# #                         p_matrix[j, i, 0] = p_matrix[i, j, 0]
# #                         val_matrix[j, i, 0] = val_matrix[i, j, 0]
# #                         if conf_matrix is not None:
# #                             conf_matrix[j, i, 0] = conf_matrix[i, j, 0]
# #                     else:
# #                         # Links not present in link_assumptions
# #                         pass

# #         # Return the values as a dictionary and store in class
# #         results = {'val_matrix': val_matrix,
# #                    'p_matrix': p_matrix,
# #                    'conf_matrix': conf_matrix}
# #         return results
# # # def _print_mci_conditions(conds_y, conds_x_lagged,
# # #                             j, i, tau, count, n_parents):
# # #     """Print information about the conditions for the MCI algorithm.

# # #     Parameters
# # #     ----------
# # #     conds_y : list
# # #         Conditions on node.
# # #     conds_x_lagged : list
# # #         Conditions on parent.
# # #     j : int
# # #         Current node.
# # #     i : int
# # #         Parent node.
# # #     tau : int
# # #         Parent time delay.
# # #     count : int
# # #         Index of current parent.
# # #     n_parents : int
# # #         Total number of parents.
# # #     """
# # #     # Remove the current parent from the conditions
# # #     conds_y_no_i = [node for node in conds_y if node != (i, tau)]
# # #     # Get the condition string for parent
# # #     condy_str = _mci_condition_to_string(conds_y_no_i)
# # #     # Get the condition string for node
# # #     condx_str = _mci_condition_to_string(conds_x_lagged)
# # #     # Formate and print the information
# # #     link_marker = {True:"o?o", False:"-?>"}
# # #     indent = "\n        "
# # #     print_str = indent + "link (%s % d) " % (var_names[i], tau)
# # #     print_str += "%s %s (%d/%d):" % (link_marker[tau==0],
# # #         var_names[j], count + 1, n_parents)
# # #     print_str += indent + "with conds_y = %s" % (condy_str)
# # #     print_str += indent + "with conds_x = %s" % (condx_str)
# # #     print(print_str)

# ###### ================================ test code =============================== #############
# # link_assumptions = None
# # tau_max = 3
# # tau_min = 1
# # # N = 10
# # remove_contemp = False
# # val_only = False
# # alpha_level = 0.05

# # M,N,t = array.shape

# # _int_link_assumptions = deepcopy(link_assumptions)
# # # Set the default selected links if none are set
# # _vars = list(range(N))
# # _lags = list(range(-(tau_max), -tau_min + 1, 1))
# # if _int_link_assumptions is None:
# #     _int_link_assumptions = {}
# #     # Set the default as all combinations
# #     for j in _vars:
# #         _int_link_assumptions[j] = {}
# #         for i in _vars:
# #             for lag in range(tau_min, tau_max + 1):
# #                 if not (i == j and lag == 0):
# #                     if lag == 0:
# #                         _int_link_assumptions[j][(i, 0)] = 'o?o'
# #                     else:
# #                         _int_link_assumptions[j][(i, -lag)] = '-?>'

# # else:

# #     if remove_contemp:
# #         for j in _int_link_assumptions.keys():
# #             _int_link_assumptions[j] = {link:_int_link_assumptions[j][link] 
# #                                 for link in _int_link_assumptions[j]
# #                                     if link[1] != 0}

# # # Make contemporaneous assumptions consistent and orient lagged links
# # for j in _vars:
# #     for link in _int_link_assumptions[j]:
# #         i, tau = link
# #         link_type = _int_link_assumptions[j][link]
# #         if tau == 0:
# #             if (j, 0) in _int_link_assumptions[i]:
# #                 if _int_link_assumptions[j][link] != _reverse_link(_int_link_assumptions[i][(j, 0)]):
# #                     raise ValueError("Inconsistent link assumptions for indices %d - %d " %(i, j))
# #             else:
# #                 _int_link_assumptions[i][(j, 0)] = _reverse_link(_int_link_assumptions[j][link])
# #         else:
# #             # Orient lagged links by time order while leaving the middle mark
# #             new_link_type = '-' + link_type[1] + '>'
# #             _int_link_assumptions[j][link] = new_link_type

# # _int_parents = _get_int_parents(parents=None)

# # val_matrix = np.zeros((N, N, tau_max + 1))
# # p_matrix = np.ones((N, N, tau_max + 1))

# # max_conds_py = N * (tau_max - tau_min + 1)
# # max_conds_px = N * (tau_max - tau_min + 1)

# # for j, i, tau, Z in _iter_indep_conds(_int_parents,
# #                                     _int_link_assumptions,
# #                                     max_conds_py,
# #                                     max_conds_px):
# #     # X = [(i, tau)]
# #     # Y = [(j, 0)]
# #     print(i,j,tau, Z)
# #     # print(i,j,abs(tau))
# #     # print(i, -abs(tau), _int_link_assumptions[j],_int_link_assumptions[j][(i, -abs(tau))])
# #     if val_only is False:
# #         # Run the independence tests and record the results
# #         if ((i, -abs(tau)) in _int_link_assumptions[j] 
# #                 and _int_link_assumptions[j][(i, -abs(tau))] in ['-->', 'o-o']):
# #             val = 1. 
# #             pval = 0.

# #         # print(val, pval)
# #         else:
# #             _,val = CMI_KNN(array,time_lag=abs(tau)).independence_measure(X=i,Y=j,Z=Z)
# #             pval = CMI_KNN(array,time_lag=abs(tau)).parallel_shuffles_significance(X=i,Y=j,Z=Z)
# #         val_matrix[i, j, abs(tau)] = val
# #         p_matrix[i, j, abs(tau)] = pval
# #     else:
# #         val = CMI_KNN(array,time_lag=abs(tau)).independence_measure(X=i,Y=j,Z=Z)
# #         val_matrix[i, j, abs(tau)] = val 
# #     print(val, pval)
# #     print(p_matrix)


# # # Threshold p_matrix to get graph
# # final_graph = p_matrix <= alpha_level 
# # print(final_graph)
# # graph = convert_to_string_graph(final_graph)

# # print(graph)
# #     # print(val, pval)  
# # # _int_link_assumptions