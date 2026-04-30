import sys, os, time, subprocess
import scipy
import numpy as np
import matplotlib.pyplot as plt

def run_no_numba(npoints, seed):
    code = f"""
import os, subprocess, time
import numpy as np
os.environ["NUMBA_DISABLE_JIT"] = "1"
from mymesh import delaunay
# pre-run in case of any first time delays
delaunay.BowyerWatson2d(np.random.rand(10,2), Constraints=np.random.choice(np.arange(0, 10, dtype=int), size=(3,2), replace=False))

np.random.seed({seed})
points = np.random.rand({npoints}, 2)
constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(3,2),replace=False)
tic = time.time()
delaunay.BowyerWatson2d(points, Constraints=constraints)
t = time.time()-tic
print(t)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    t = float(result.stdout)

    return t


from mymesh import delaunay
npoints = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=int)

bowyer_watson_avg = []
bowyer_watson_std = []
bowyer_watson_no_numba_avg = []
bowyer_watson_no_numba_std = []
triangle_avg = []
triangle_std = []

# pre-run for any first time pre-compiling
points = np.random.rand(10,2)
constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(5,2),replace=False)
delaunay.BowyerWatson2d(points, Constraints=constraints)
delaunay.Triangle(points, Constraints=constraints)
#
N = 3
for i,n in enumerate(npoints):
    # BowyerWatson
    if i < 2 or bowyer_watson_avg[-1] < 5:
        reps = []
        for j in range(N):
            np.random.seed(j)
            points = np.random.rand(n, 2)
            constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(5,2),replace=False)
            tic = time.time()
            delaunay.BowyerWatson2d(points, Constraints=constraints)
            reps.append(time.time()-tic)

        bowyer_watson_avg.append(np.mean(reps))
        bowyer_watson_std.append(np.std(reps))

    # BowyerWatson - No Numba
    if i < 2 or bowyer_watson_no_numba_avg[-1] < 5:
        reps = []
        for j in range(N):
            t = run_no_numba(n, j)
            reps.append(t)
        bowyer_watson_no_numba_avg.append(np.mean(reps)/N)
        bowyer_watson_no_numba_std.append(np.std(reps)/N)


    # Triangle
    if i < 2 or triangle_avg[-1] < 5:
        reps = []
        for j in range(N):
            np.random.seed(j)
            points = np.random.rand(n, 2)
            constraints = np.random.choice(np.arange(0, len(points), dtype=int), size=(5,2),replace=False)
            tic = time.time()
            delaunay.Triangle(points, Constraints=constraints)
            reps.append(time.time()-tic)
        triangle_avg.append(np.mean(reps))
        triangle_std.append(np.std(reps))

# Plot
plt.errorbar(npoints[:len(bowyer_watson_avg)], bowyer_watson_avg, yerr=bowyer_watson_std, color='#bf616a')
plt.errorbar(npoints[:len(bowyer_watson_no_numba_avg)], bowyer_watson_no_numba_avg, yerr=bowyer_watson_no_numba_std, color='#5e81ac')
plt.errorbar(npoints[:len(triangle_avg)], triangle_avg, yerr=triangle_std, color='#ebcb8b', linestyle='dotted')
plt.xscale('log')
plt.yscale('log')
plt.legend(['mymesh', 'mymesh (no numba)', 'Triangle'])
plt.xlabel('# of points')
plt.ylabel('Time (s)')
plt.title('2D Constrained Delaunay Triangulation')
plt.grid()
plt.show()