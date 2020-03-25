#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'Cython')
get_ipython().run_line_magic('load_ext', 'line_profiler')
import numpy
import itertools
import seaborn
import pandas
import statsmodels
import statsmodels.api


# In[2]:


get_ipython().run_cell_magic('cython', '', '# cython: linetrace=True\n# cython: binding=True\n# distutils: define_macros=CYTHON_TRACE_NOGIL=1\nimport numpy\nimport cython\nimport random\nfrom cython.parallel import prange\ncimport numpy as cnumpy\ncimport cython\n\n@cython.nonecheck(False)\n@cython.boundscheck(False)\ndef get_permutations(long[:,:] cb_ids, long[:] sites, long[:] sub_branches, double[:] p, long niter):\n    cdef cnumpy.ndarray[cnumpy.uint8_t, ndim = 3, cast=True] ps\n    cdef cnumpy.ndarray[cnumpy.uint8_t, ndim = 2, cast=True] sub_bool_array\n    cdef long[:,:] num_shared_sub = numpy.zeros(shape=(cb_ids.shape[0], niter), dtype=numpy.long)\n    cdef long[:] num_shared_sub2 = numpy.zeros(shape=(niter), dtype=numpy.long)\n    cdef long size\n    cdef long prev\n    cdef long[:] site_indices\n    cdef Py_ssize_t i,j,s,b # Py_ssize_t is the proper C type for Python array indices.\n    ps = numpy.zeros(shape=(sub_branches.shape[0], niter, sites.shape[0]), dtype=numpy.bool_)\n    for i in range(sub_branches.shape[0]):\n        size = sub_branches[i]\n        if size!=0:\n            if size in sub_branches[:i]:\n                prev = numpy.arange(i)[numpy.equal(sub_branches[:i], size)][0]\n                ps[i,:,:] = ps[prev,numpy.random.permutation(numpy.arange(ps.shape[1])),:]\n            else:\n                for j in range(niter):\n                    site_indices = numpy.random.choice(a=sites, size=size, replace=False, p=p)\n                    ps[i,j,site_indices] = True\n    for i in range(cb_ids.shape[0]):\n        for b in range(cb_ids[i,:].shape[0]):\n            if cb_ids[i,b]==cb_ids[i,0]:\n                sub_bool_array = ps[cb_ids[i,b],:,:].copy()\n            else:\n                sub_bool_array *= ps[cb_ids[i,b],:,:]\n        num_shared_sub2 = sub_bool_array.sum(axis=1)\n        num_shared_sub[i,:] = num_shared_sub2\n    return numpy.asarray(num_shared_sub)')


# In[15]:


sites = numpy.arange(100)
sub_branches = numpy.random.choice([0,5,10,20], 100)
p = numpy.arange(sites.shape[0])+10
p = p/p.sum()
niter = 100
cb_ids = numpy.array(list(itertools.combinations(numpy.arange(sub_branches.shape[0]), 2)), dtype=numpy.int64)

hoge = get_permutations(cb_ids, sites, sub_branches, p, niter)


# In[27]:


sites = numpy.arange(100)
sub_branches = numpy.random.choice([0,5,10,20], 100)
p = numpy.arange(sites.shape[0])+10
p = p/p.sum()
niter = 100
cb_ids = numpy.array(list(itertools.combinations(numpy.arange(sub_branches.shape[0]), 2)), dtype=numpy.int64)

get_ipython().run_line_magic('timeit', 'hoge = get_permutations(cb_ids, sites, sub_branches, p, niter)')
#%timeit hoge = get_permutations2(cb_ids, sites, sub_branches, p, niter)
get_ipython().run_line_magic('timeit', 'hoge = get_permutations(cb_ids, sites, sub_branches, p, niter)')
#%timeit hoge = get_permutations2(cb_ids, sites, sub_branches, p, niter)
get_ipython().run_line_magic('timeit', 'hoge = get_permutations(cb_ids, sites, sub_branches, p, niter)')
#%timeit hoge = get_permutations2(cb_ids, sites, sub_branches, p, niter)


# In[ ]:





# In[49]:


L = numpy.arange(4) # stone labels (0,1,2,3)
MN = numpy.array([2,3]) # M and N
cb_ids = numpy.expand_dims(numpy.arange(MN.shape[0]), axis=0)
p = numpy.array([0.1,0.2,0.3,0.4,]) # stone frequencies
#p = numpy.array([0.25,0.25,0.25,0.25]) # stone frequencies
niter = 10000 # Number of permutations

# Kamesan method
X = sum([ numpy.array(probs).prod() for probs in list(itertools.combinations(p, MN[0])) ])
Y = sum([ numpy.array(probs).prod() for probs in list(itertools.combinations(p, MN[1])) ])
Z = X * Y
P_k1 = (1/Z)*(3*p.prod())*(p.sum())
print('Kamesan: Probability of k={} is {}'.format(1,P_k1))

# Simulation
out = get_permutations(cb_ids, L, MN, p, niter)
for i in [0,1,2,3,4]:
    prob = (out==i).sum()/niter
    print('Permutation: Probability of k={} is {}'.format(i,prob))


# In[28]:


def get_permutations2_days_sentinel():
    pass

get_ipython().run_line_magic('lprun', '-f get_permutations2 get_permutations2(cb_ids, sites, sub_branches, p, niter)')


# In[29]:


dfq = get_permutations(cb_ids, sites, sub_branches+50, p, niter=1000)
seaborn.distplot(dfq[0,:])


# In[30]:


import scipy
fit_alpha, fit_loc, fit_beta = scipy.stats.gamma.fit(dfq[0,:])
print(fit_alpha, fit_loc, fit_beta)


# In[37]:


a = numpy.random.choice([0,10,20,30], 1000)
v = 30
(a>=v).sum() / len(a)


# In[56]:


x = 'x'
y = 'y'
df = pandas.DataFrame({y:dfq[0,:], x:numpy.ones_like(dfq[0,:])})
glm_formula = y+" ~ "+x
reg_family = family=statsmodels.api.families.Binomial(link=statsmodels.api.families.links.log)
mod = statsmodels.formula.api.glm(formula=glm_formula, data=df, family=reg_family)
res = mod.fit()
res


# In[57]:


res.summary()


# In[52]:


res.summary()


# In[ ]:




