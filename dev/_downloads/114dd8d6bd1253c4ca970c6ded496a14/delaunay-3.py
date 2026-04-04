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
delaunay.ConvexHull(np.random.rand(10,2), method='GiftWrapping')
np.random.seed({seed})
points = np.random.rand({npoints}, 2)
tic = time.time()
delaunay.ConvexHull(points, method='GiftWrapping')
t = time.time()-tic
print(t)
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True)
    t = float(result.stdout)

    return t


from mymesh import delaunay
npoints = np.array([1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7], dtype=int)

gift_wrapping_avg = []
gift_wrapping_std = []
gift_wrapping_no_numba_avg = []
gift_wrapping_no_numba_std = []
quickhull_avg = []
quickhull_std = []
scipy_avg = []
scipy_std = []

# pre-run for any first time pre-compiling
points = np.random.rand(10,2)
delaunay.ConvexHull(points, method='GiftWrapping')
delaunay.ConvexHull(points, method='QuickHull')
delaunay.ConvexHull(points, method='scipy')

N = 3
for i,n in enumerate(npoints):
    # QuickHull
    if i < 2 or quickhull_avg[-1] < 5:
        reps = []
        for j in range(N):
            np.random.seed(j)
            points = np.random.rand(n, 2)
            tic = time.time()
            delaunay.ConvexHull(points, method='QuickHull')
            reps.append(time.time()-tic)

        quickhull_avg.append(np.mean(reps))
        quickhull_std.append(np.std(reps))
    # GiftWrapping
    if i < 2 or gift_wrapping_avg[-1] < 5:
        reps = []
        for j in range(N):
            np.random.seed(j)
            points = np.random.rand(n, 2)
            tic = time.time()
            delaunay.ConvexHull(points, method='GiftWrapping')
            reps.append(time.time()-tic)

        gift_wrapping_avg.append(np.mean(reps))
        gift_wrapping_std.append(np.std(reps))

    # GiftWrapping - No Numba
    if i < 2 or gift_wrapping_no_numba_avg[-1] < 5:
        reps = []
        for j in range(N):
            t = run_no_numba(n, j)
            reps.append(t)
        gift_wrapping_no_numba_avg.append(np.mean(reps)/N)
        gift_wrapping_no_numba_std.append(np.std(reps)/N)

    # SciPy
    if i < 2 or scipy_avg[-1] < 5:
        reps = []
        for j in range(N):
            np.random.seed(j)
            points = np.random.rand(n, 2)
            tic = time.time()
            delaunay.ConvexHull(points, method='scipy')
            reps.append(time.time()-tic)
        scipy_avg.append(np.mean(reps))
        scipy_std.append(np.std(reps))


# Plot
plt.errorbar(npoints[:len(quickhull_avg)], quickhull_avg,
                        yerr=quickhull_std, color='blue')
plt.errorbar(npoints[:len(gift_wrapping_avg)], gift_wrapping_avg,
                        yerr=gift_wrapping_std, color='red')
plt.errorbar(npoints[:len(gift_wrapping_no_numba_avg)],
                gift_wrapping_no_numba_avg,
                yerr=gift_wrapping_no_numba_std, color='pink')
plt.errorbar(npoints[:len(scipy_avg)], scipy_avg,
            yerr=scipy_std, color='green')
plt.xscale('log')
plt.yscale('log')
plt.legend(['mymesh QuickHull', 'mymesh GiftWrapping', 'mymesh GiftWrapping (no numba)', 'scipy (qhull)'])
plt.xlabel('# of points')
plt.ylabel('Time (s)')
plt.title('2D Convex Hull')
plt.grid()
plt.show()