#!/usr/bin/env python

import IPython
import gc
import sys
import time
import numpy as np
import scipy.io
import scipy.sparse as sp
import scipy.spatial.distance
from sklearn.feature_extraction.text import TfidfTransformer
import sklearn.utils
from collections import Counter
import struct
try:
    import numexpr as ne
    have_numexpr = True
except ImportError:
    have_numexpr = False

MAX_BIGRAM = 2**16

f = open(sys.argv[1])
ulong_size = struct.unpack("<i", f.read(4))[0]
ulong_fmt = '<u%d' % ulong_size
FMT = "%%0%dx" % (ulong_size*2)

rec_hdr = np.dtype( [ ('caller', ulong_fmt), ('pc', ulong_fmt), ('cr3', ulong_fmt), ('nbins', '<I4') ] )
hist_entry = [ ('key', '<H'), ('value', '<u4') ]

meta = []
data = []
rows = []
cols = []

print >>sys.stderr, "Parsing file..."
i = 0
while True:
    hdr = np.fromfile(f, dtype=rec_hdr, count=1)
    if not hdr: break
    entries = np.fromfile(f, dtype=hist_entry, count=hdr['nbins'])
    # Might happen if a tap only wrote one byte. In that case there's no bigram
    if entries.size == 0: continue
    #if len(entries) < 5: continue
    #print >>sys.stderr, "Parsed entry with %d bins, file offset=%d" % (hdr['nbins'],f.tell())
    cols.extend(entries['key'])
    rows.extend([i]*len(entries))
    data.extend(entries['value'])
    meta.append(hdr)

    i += 1
    if i == 8192: break

f.close()

print >>sys.stderr, "Parsed", i, "tap points"

print >>sys.stderr, "Converting to nparrays..."
data = np.array(data,dtype=np.float64)
rows = np.array(rows)
cols = np.array(cols)
meta = np.array(meta)

print >>sys.stderr, "Creating sparse matrix..."
spdata = sp.coo_matrix((data,[rows,cols]), (i, MAX_BIGRAM), dtype=np.float64)
#spdata = {}
#for i in xrange(len(data)):
#    if i % 10000 == 0: print >>sys.stderr, i,"/",len(data)
#    for k,v in data[i]:
#        spdata[i,k] = v

print >>sys.stderr, "Converting to CSR format..."
spdata = spdata.tocsr()

print >>sys.stderr, "Normalizing..."
row_sums = np.array(spdata.sum(axis=1))[:,0]
row_indices, col_indices = spdata.nonzero()
spdata.data /= row_sums[row_indices]

data = spdata.todense()

num_means = int(sys.argv[2])
centroids = np.empty((num_means, MAX_BIGRAM), dtype=np.float64)

for i in range(num_means):
    centroids[i] = data[np.random.randint(len(data))]
    while any( np.all(centroids[i] == centroids[j]) for j in range(i) ):
        centroids[i] = data[np.random.randint(len(data))]

means_idx = [6920, 206, 5956, 3537, 3710, 7754, 4540, 3167, 4897, 727]
print >>sys.stderr, "Means chosen:", means_idx
centroids = data[means_idx]
#centroids = data[[47,5]]
#print centroids

#IPython.embed()

assignments = np.zeros((data.shape[0],))
iteration = 0
while True:
    print >>sys.stderr, "Iteration", iteration
    iteration += 1
    dists = scipy.spatial.distance.cdist(centroids, data, 'euclidean')
    dists = dists.T
    new_assignments = np.argmin(dists, axis=1)
    if np.all(new_assignments == assignments):
        print >>sys.stderr, "done"
        break
    print >>sys.stderr, "%d assignment(s) changed this iteration" % np.sum(assignments != new_assignments)
    assignments = new_assignments
    print >>sys.stderr, "RSS = %e" % np.sum(dists[[np.arange(data.shape[0]),assignments]]**2)
    rss = 0.0
    for i in range(data.shape[0]):
        c = assignments[i]
        x = dists[i,assignments[i]]
        rss += x*x

    for i in range(num_means):
        clust = data[assignments == i]
        centroids[i] = np.sum(clust, axis=0) / len(clust)

for i in range(data.shape[0]):
    print assignments[i]
