"""Objective functions for the IBO experiments."""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import RegularGridInterpolator

__all__ = ["kim1", "f_egg", "get_benchmarks", "OFFSETS", "shift_benchmarks",
           "sample_gp_function_", "generate_centers"]


# =====================================================================
#  1. Kim1
# =====================================================================
def kim1(X):
    """
    The Kim1 function of Kim (2020), Eq. 27, on [-16, 16]^2.

    Variables:
    - X: (n, 2) array of points.

    A sinusoidal ripple on an asymmetric quadratic bowl, non-convex with many
    local minima of comparable depth.  Its global minimum is -1.9715 at
    (4.7213, 3.1709).
    """
    X = np.atleast_2d(np.asarray(X, float))
    x1, x2 = X[:, 0], X[:, 1]
    return (np.sin(x1) + np.cos(x2) + 0.016 * (x1 - 5) ** 2
            + 0.008 * (x2 - 5) ** 2)


f_egg = kim1          # the name used in earlier versions of the code


# =====================================================================
#  2. Ten stationary benchmarks on [-1, 1]^2
# =====================================================================
def get_benchmarks():
    """
    The ten stationary benchmarks, each rescaled to [-1, 1]^2 and shifted so
    that its minimum value is exactly 1.

    Returns a dict of functions and a dict of the minimiser locations, both
    keyed by name.  Only the value is normalised here; `shift_benchmarks`
    moves the locations.
    """
    f_bench = {}
    target_argmins = {}

    def scale_input(x, low, high):
        # Maps [-1, 1] to [low, high]
        return low + (x + 1.0) * (high - low) / 2.0

    def scale_argmin(orig_argmin, low, high):
        # Maps original point back to [-1, 1] for the target dict
        return 2.0 * (orig_argmin - low) / (high - low) - 1.0

    # Helper decorator to vectorize functions over last axis
    def vectorize_2d(func):
        def wrapper(XY):
            XY = np.asarray(XY)
            # If XY is 1D, make it 2D
            if XY.ndim == 1:
                return func(XY)
            # Reshape to (-1, 2) for computation
            shape_orig = XY.shape[:-1]
            XY_flat = XY.reshape(-1, XY.shape[-1])
            Z_flat = np.array([func(pt) for pt in XY_flat])
            return Z_flat.reshape(shape_orig)
        return wrapper

    # 1. Ackley
    @vectorize_2d
    def ackley(xx):
        xs = scale_input(xx, -32.768, 32.768)
        d = xs.shape[0]
        sum1 = np.sum(xs**2)
        sum2 = np.sum(np.cos(2 * np.pi * xs))
        y = -20 * np.exp(-0.2 * np.sqrt(sum1 / d)) - np.exp(sum2 / d) + 20 + np.exp(1)
        return y + 1.0
    f_bench["Ackley"] = ackley
    target_argmins["Ackley"] = np.array([0.0, 0.0])

    # 2. Schwefel
    @vectorize_2d
    def schwefel(xx):
        xs = scale_input(xx, -500, 500)
        d = xs.shape[0]
        y = 418.9829 * d - np.sum(xs * np.sin(np.sqrt(np.abs(xs))))
        return y + 1.0
    f_bench["Schwefel"] = schwefel
    target_argmins["Schwefel"] = np.array([scale_argmin(420.9687, -500, 500)] * 2)

    # 3. Rastrigin
    @vectorize_2d
    def rastrigin(xx):
        xs = scale_input(xx, -5.12, 5.12)
        d = xs.shape[0]
        y = 10 * d + np.sum(xs**2 - 10 * np.cos(2 * np.pi * xs))
        return y + 1.0
    f_bench["Rastrigin"] = rastrigin
    target_argmins["Rastrigin"] = np.array([0.0, 0.0])

    # 4. Levy
    @vectorize_2d
    def levy(xx):
        xs = scale_input(xx, -10, 10)
        w = 1 + (xs - 1) / 4
        term1 = np.sin(np.pi * w[0])**2
        term3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
        wi = w[:-1]
        sum_term = np.sum((wi - 1)**2 * (1 + 10 * np.sin(np.pi * wi + 1)**2))
        return term1 + sum_term + term3 + 1.0
    f_bench["Levy"] = levy
    target_argmins["Levy"] = np.array([scale_argmin(1.0, -10, 10)] * 2)

    # 5. Beale
    @vectorize_2d
    def beale(xx):
        xs = scale_input(xx, -4.5, 4.5)
        x, y = xs
        val = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
        return val + 1.0
    f_bench["Beale"] = beale
    target_argmins["Beale"] = np.array([scale_argmin(3.0, -4.5, 4.5),
                                        scale_argmin(0.5, -4.5, 4.5)])

    # 6. Dixon-Price
    @vectorize_2d
    def dixon_price(xx):
        xs = scale_input(xx, -10, 10)
        d = len(xs)
        term1 = (xs[0] - 1)**2
        ii = np.arange(2, d + 1)
        sum_term = np.sum(ii * (2 * xs[1:]**2 - xs[:-1])**2)
        return term1 + sum_term + 1.0
    f_bench["Dixon-Price"] = dixon_price
    x1_orig = 1.0
    x2_orig = 2**(-(2**2 - 2)/2**2)
    target_argmins["Dixon-Price"] = np.array([scale_argmin(x1_orig, -10, 10),
                                              scale_argmin(x2_orig, -10, 10)])

    # 7. Griewank
    @vectorize_2d
    def griewank(xx):
        xs = scale_input(xx, -600, 600)
        d = len(xs)
        ii = np.arange(1, d + 1)
        y = np.sum(xs**2 / 4000) - np.prod(np.cos(xs / np.sqrt(ii))) + 1
        return y + 1.0
    f_bench["Griewank"] = griewank
    target_argmins["Griewank"] = np.array([0.0, 0.0])

    # 8. Goldstein-Price
    @vectorize_2d
    def goldpr(xx):
        xs = scale_input(xx, -2, 2)
        x1, x2 = xs
        fact1 = 1 + (x1+x2+1)**2 * (19-14*x1+3*x1**2-14*x2+6*x1*x2+3*x2**2)
        fact2 = 30 + (2*x1-3*x2)**2 * (18-32*x1+12*x1**2+48*x2-36*x1*x2+27*x2**2)
        return (fact1 * fact2 - 3.0) + 1.0
    f_bench["Goldstein-Price"] = goldpr
    target_argmins["Goldstein-Price"] = np.array([scale_argmin(0.0, -2, 2),
                                                  scale_argmin(-1.0, -2, 2)])

    # 9. Bohachevsky #3
    @vectorize_2d
    def boha3(xx):
        xs = scale_input(xx, -100, 100)
        x1, x2 = xs
        y = x1**2 + 2*x2**2 - 0.3*np.cos(3*np.pi*x1 + 4*np.pi*x2) + 0.3
        return y + 1.0
    f_bench["Bohachevsky"] = boha3
    target_argmins["Bohachevsky"] = np.array([0.0, 0.0])

    # 10. Matyas
    @vectorize_2d
    def matyas(xx):
        xs = scale_input(xx, -100, 100)
        x1, x2 = xs
        y = 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
        return y + 1.0
    f_bench["Matyas"] = matyas
    target_argmins["Matyas"] = np.array([0.0, 0.0])

    return f_bench, target_argmins


# Five of the ten benchmarks have their minimiser at the origin, which is also
# where the fixed grid of IBO_4 cuts the domain: the refinement step then
# reaches the optimum by projection rather than by search.  These offsets move
# every minimiser off the grid lines, and none of them is a dyadic fraction,
# so no cut of a tree of depth three lands on one either.
OFFSETS = {
    "Ackley":          np.array([ 0.137, -0.291]),
    "Schwefel":        np.array([-0.213, -0.169]),
    "Rastrigin":       np.array([-0.317,  0.143]),
    "Levy":            np.array([ 0.191, -0.237]),
    "Beale":           np.array([-0.163,  0.211]),
    "Dixon-Price":     np.array([ 0.229, -0.187]),
    "Griewank":        np.array([ 0.271,  0.159]),
    "Goldstein-Price": np.array([-0.141,  0.223]),
    "Bohachevsky":     np.array([ 0.183, -0.249]),
    "Matyas":          np.array([-0.277,  0.131]),
}


def shift_benchmarks(f_bench, target_argmins, offsets=OFFSETS):
    """Return (functions, argmins) with each minimiser displaced by its offset.

    Shifting the input leaves the minimum *value* unchanged, so every function
    still has minimum exactly 1; only the location moves.

    Apply this before fitting the length-scales, so that the kernel of a
    benchmark is estimated on the function actually optimised.
    """
    def make(fn, delta):
        def g(XY):
            XY = np.asarray(XY, float)
            return fn(XY - delta)
        return g

    out_f, out_a = {}, {}
    for name, fn in f_bench.items():
        delta = offsets[name]
        out_f[name] = make(fn, delta)
        out_a[name] = np.asarray(target_argmins[name], float) + delta
    return out_f, out_a


# =====================================================================
#  3. Inhomogeneous surfaces
# =====================================================================
def sample_gp_function_(centers, seed=7):
    """
    Draw a surface from a non-stationary Gaussian process on [0, 1]^2.

    Variables:
    - centers: (3, 2) array of anchor points, in the order rough, smooth,
      intermediate.  The single surface of the paper uses
      [[0.2, 0.2], [0.8, 0.8], [0.5, 0.5]].
    - seed: fixes the draw.

    Three Matérn components of smoothness 1/2, 5/2 and 3/2 are mixed by
    Gaussian weights normalised in l2, so that the pointwise variance is
    constant and the zones differ only in correlation structure.  The surface
    is shifted so that its minimum is exactly 1 and returned as an
    interpolator on an 80 x 80 grid, together with the grid and its values.
    """
    rng = np.random.default_rng(seed)

    # Domain grid
    n1 = 80
    x_line = np.linspace(0, 1, n1)
    X_m, Y_m = np.meshgrid(x_line, x_line, indexing='ij')
    xs = np.column_stack([X_m.ravel(), Y_m.ravel()])
    nxs = xs.shape[0]

    # Distance matrix
    D = cdist(xs, xs)

    # --- Kernels ---
    K_rough = np.exp(-D / 0.2)

    sqrt3_D = np.sqrt(3) * D
    K_med = (1 + sqrt3_D / 0.4) * np.exp(-sqrt3_D / 0.4)

    K_smooth = (1 + np.sqrt(5) * D / 0.3 + (5 * D**2) / (3 * 0.3**2)) \
        * np.exp(-np.sqrt(5) * D / 0.3)

    # --- Spatial weights with movable centers ---
    def get_weights(X, centers):
        c1, c2, c3 = centers

        w1 = np.exp(-((X[:, 0]-c1[0])**2 + (X[:, 1]-c1[1])**2) / 0.1)
        w2 = np.exp(-((X[:, 0]-c2[0])**2 + (X[:, 1]-c2[1])**2) / 0.1)
        w3 = np.exp(-((X[:, 0]-c3[0])**2 + (X[:, 1]-c3[1])**2) / 0.1)

        total = np.sqrt(w1**2 + w2**2 + w3**2) + 1e-6
        return w1/total, w2/total, w3/total

    b1, b2, b3 = get_weights(xs, centers)

    # Non-stationary covariance
    K = (
        np.outer(b1, b1) * K_rough +
        np.outer(b2, b2) * K_smooth +
        np.outer(b3, b3) * K_med
    )

    K += np.eye(nxs) * 1e-8

    L = np.linalg.cholesky(K)
    zs = L @ rng.standard_normal(nxs)
    Z2 = zs.reshape(n1, n1)
    Z2 = Z2 - Z2.min() + 1.0

    f_interp = RegularGridInterpolator((x_line, x_line), Z2,
                                       bounds_error=False, fill_value=None)

    return f_interp, x_line, Z2


def generate_centers(min_dist=0.3, rng=None):
    """
    Draw three anchor points in [0.1, 0.9]^2, no two closer than `min_dist`.

    Pass a seeded Generator as `rng`, and draw the anchors and the surface
    seed from the same one, so that a batch of surfaces is reproducible.  The
    number of rejected candidates depends on `min_dist`, so changing it
    changes every surface drawn afterwards.
    """
    rng = rng or np.random
    while True:
        centers = rng.uniform(0.1, 0.9, (3, 2))
        pairwise_d = cdist(centers, centers)
        np.fill_diagonal(pairwise_d, np.inf)
        if np.all(pairwise_d >= min_dist):
            return centers
