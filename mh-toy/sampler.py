import numpy as np
import random as rand
import matplotlib.pyplot as plt



def accept_reject(u, p_sample, scores, tau):
	'''
	return 1 if accept, return 0 if reject.
	'''
	[p_prev, p_new] = p_sample
	[score_prev, score_new] = scores

	accept_threshold = np.exp(score_new/tau)/ np.exp(score_prev/tau)
	accept_threshold *=  p_prev/p_new
	if u <= accept_threshold:# or accept_threshold>=1 :
		return 1
	return 0

def f_score(sample):
	'''
	dummy score function. (modify)
	'''
	score=1.0 
	return score

def f_sampler():
	'''
	dummy sampler function. (modify)
	'''
	sample=''
	return sample

n_iter, n_a, burn_in, n_sample = 0,0, 10000, 100000   # num of samples
tau = 0.8
p_prev, p_new = 1.0, 0.7
score_prev, score_new=1.0, 0.7
sample_list=[]
# rand.seed(a=1024)
hyp=''
# p_prev # evaluate prob of y* given the proposal distribution \theta_pre
while n_a < n_sample:
	#hyp, p_new = # generate hypothesis 
	hyp = f_sampler()
	score_new= f_score(hyp)
	#score_prev, score_new=  f_score(y_star),f_score(hyp) #get score
	u = rand.uniform(1,0)
	if accept_reject(u, [p_prev, p_new], [score_prev, score_new], tau):
		sample_list.append(hyp)
		# update prev samples to be current.
		p_prev= p_new
		score_prev= score_new
		n_a+=1
	n_iter+=1
print("DONE", '\tn_iter:', n_iter, '\tn_a:',n_a)


# def accept_reject(u, p_sample, scores, tau):
# 	'''
# 	return 1 if accept, return 0 if reject.
# 	'''
# 	[p_prev, p_new] = p_sample
# 	[score_prev, score_new] = scores

# 	accept_threshold = score_new/ score_prev
# 	accept_threshold *=  p_prev/p_new
# 	# print('accept_threshold', accept_threshold, '\tbool:\t', accept_threshold>=1)
# 	if u <= accept_threshold or accept_threshold>=1 :
# 		return 1
# 	return 0


# def normal_prob(x,mu,sigma):
# 	return np.exp(-(x-mu)*(x-mu)/(2*sigma*sigma) )/np.sqrt(2*np.pi*sigma*sigma)


# ### options for samples & proposal distribution ####
# n_iter, n_a, burn_in, n_sample = 0,0, 1000, 10000   # num of samples
# mu_p, sigma_prop=0.0, np.sqrt(100) # for normal distribution
# u_low, u_high = -100, 100 # for uniform
# mu_t, sigma_true=2.0, np.sqrt(80) # for true distribution
# proposal_dist = 'unif' # whether to get normal or unif
# tau=0.8 # not used here
# #######################################################

# sample_list=[] # accepted sample list
# hyp=0 # initial hypothesis

# # sample
# if proposal_dist=='unif':
# 	prop_samples = np.random.uniform(u_low,u_high,n_sample)
# else:
# 	prop_samples = np.random.normal(mu_p,sigma_prop,n_sample)
# print("===================================================")
# print("mean prop_samples", np.mean(prop_samples))
# print("std of prop_samples", np.sqrt(np.var(prop_samples)))
# # plot
# fig0, ax0 = plt.subplots()
# count, bins, ignored = ax0.hist(prop_samples, 30, normed=True)
# ax0.plot(bins, 1/(sigma_prop * np.sqrt(2 * np.pi)) *
# 		np.exp( - (bins - mu_p)**2 / (2 * sigma_prop**2) ),
# 		linewidth=2, color='r')
# plt.savefig("proposed.pdf")
# print("===================================================")
# if proposal_dist=='unif':
# 	p_prev = 1/(u_high-u_low)
# else:
# 	p_prev= normal_prob(hyp,mu_p,sigma_prop)
# score_prev= normal_prob(hyp,mu_t,sigma_true)
# print('p_prev: ', p_prev,'\t score_prev: ', score_prev)
# p_prev # evaluate prob of y* given the proposal distribution \theta_pre
# for hyp in prop_samples:
# 	if proposal_dist=='unif':
# 		p_new = 1/(u_high-u_low)
# 	else:
# 		p_new = normal_prob(hyp,mu_p,sigma_prop)
# 	score_new = normal_prob(hyp,mu_t,sigma_true)
# 	#hyp, p_new = # generate hypothesis 
# 	#score_prev, score_new=  f_score(y_star),f_score(hyp) #get score
# 	u = rand.uniform(1,0)
# 	if n_iter>burn_in and accept_reject(u, [p_prev, p_new], [score_prev, score_new], tau):
# 		sample_list.append(hyp)
# 		n_a+=1
# 		p_prev= p_new
# 		score_prev= score_new
# 		# print('n_a:', n_a )
# 	n_iter+=1
# print("===================================================")
# print("DONE", '\tn_iter:', n_iter, '\tn_a:',n_a, '\tacceptance ratio:', n_a/(n_iter+0.0))
# print("mean accepted_samples", np.mean(sample_list))
# print("median accepted_samples", np.mean(sample_list))
# print("std of sample_list", np.sqrt(np.var(sample_list)))
# print("===================================================")

# fig1, ax1 = plt.subplots()
# count, bins, ignored = ax1.hist(sample_list, 30, normed=True)
# ax1.plot(bins, 1/(sigma_true * np.sqrt(2 * np.pi)) *
# 		np.exp( - (bins - mu_t)**2 / (2 * sigma_true**2) ),
# 		linewidth=2, color='r')
# plt.savefig("accepted.pdf")
