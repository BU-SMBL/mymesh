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
delaunay.BowyerWatson3d(np.random.rand(10,3))
np.random.seed({seed})
points = np.random.rand({npoints}, 3)
tic = time.time()
delaunay.BowyerWatson3d(points)
t = time.time()-tic
print(t)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    t = float(result.stdout)

    return t


from mymesh import delaunay
npoints = np.array([1e1, 1e2, 1e3, 1e4], dtype=int)

bowyer_watson_avg = []
bowyer_watson_std = []
bowyer_watson_no_numba_avg = []
bowyer_watson_no_numba_std = []
scipy_avg = []
scipy_std = []
tetgen_avg = []
tetgen_std = []

# pre-run for any first time pre-compiling
points = np.random.rand(10,3)
delaunay.BowyerWatson3d(points)
scipy.spatial.Delaunay(points)
# delaunay.TetGen(points)

N = 3
for i,n in enumerate(npoints):
    # BowyerWatson
    if i < 2 or bowyer_watson_avg[-1] < 5:
        reps = []
        for j in range(N):
            np.random.seed(j)
            points = np.random.rand(n, 3)
            tic = time.time()
            delaunay.BowyerWatson3d(points)
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

    # SciPy
    if i < 2 or scipy_avg[-1] < 5:
        reps = []
        for j in range(N):
            np.random.seed(j)
            points = np.random.rand(n, 3)
            tic = time.time()
            scipy.spatial.Delaunay(points)
            reps.append(time.time()-tic)
        scipy_avg.append(np.mean(reps))
        scipy_std.append(np.std(reps))

    # TetGen
    # if i < 2 or tetgen_avg[-1] < 5:
    #     reps = []
    #     for j in range(N):
    #         np.random.seed(j)
    #         points = np.random.rand(n, 3)
    #         tic = time.time()
    #         delaunay.Triangle(points)
    #         reps.append(time.time()-tic)
    #     tetgen_avg.append(np.mean(reps))
    #     tetgen_std.append(np.std(reps))

# Plot
plt.errorbar(npoints[:len(bowyer_watson_avg)], bowyer_watson_avg, yerr=bowyer_watson_std, color='#bf616a')
plt.errorbar(npoints[:len(bowyer_watson_no_numba_avg)], bowyer_watson_no_numba_avg, yerr=bowyer_watson_no_numba_std, color='#5e81ac')
plt.errorbar(npoints[:len(scipy_avg)], scipy_avg, yerr=scipy_std, color='#a3be8c', linestyle='dotted')
# plt.errorbar(npoints[:len(tetgen_avg)], tetgen_avg, yerr=tetgen_std, color='#ebcb8b', linestyle='dotted')
plt.xscale('log')
plt.yscale('log')
plt.legend(['mymesh', 'mymesh (no numba)', 'scipy (qhull)'])#, 'TetGen'])
plt.xlabel('# of points')
plt.ylabel('Time (s)')
plt.title('3D Delaunay Triangulation')
plt.grid()
plt.show()