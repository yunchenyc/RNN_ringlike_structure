import numpy as np
from scipy import optimize
from scipy.spatial.distance import pdist, squareform

# @title @getting_fitted_ellipses
# ref: https://blog.csdn.net/zhuzi2129/article/details/106520105
# ref: https://blog.csdn.net/weixin_39849839/article/details/108313284

## fit surface
def fit_plane1(p, xi, yi, zi):
    aa, bb, cc, dd = p
    # eq = ((aa*xi + bb*yi + cc*zi + dd)**2/(aa**2 + bb**2 + cc**2)).mean()
    eq = (np.sqrt((aa ** 2 * xi + aa * (bb * yi + cc * zi + dd)) ** 2
                 + (bb ** 2 * yi + bb * (aa * xi + cc * zi + dd)) ** 2
                 + (cc ** 2 * zi + cc * (aa * xi + bb * yi + dd)) ** 2) 
          / (aa ** 2 + bb ** 2 + cc ** 2))
    return eq


def fit_plane2(p, xi, yi):
    aa, bb, cc = p
    eq = aa * xi + bb * yi + cc
    return eq


def fit_planes(x, y, z):
    # not parallel
    dictplanep = {}

    ls_p = optimize.least_squares(fit_plane1, np.ones((4,)), args=(x, y, z))
    dictplanep['titled'] = ls_p.x

    ls_p2 = optimize.least_squares(fit_plane2, np.ones((3,)), args=(x, y))
    dictplanep['paralleltoz'] = [ls_p2.x[0], ls_p2.x[1], 0, ls_p2.x[2]]

    ls_p3 = optimize.least_squares(fit_plane2, np.ones((3,)), args=(x, z))
    dictplanep['paralleltoy'] = [ls_p3.x[0], 0, ls_p3.x[1], ls_p2.x[2]]

    ls_p4 = optimize.least_squares(fit_plane2, np.ones((3,)), args=(y, z))
    dictplanep['paralleltox'] = [0, ls_p4.x[0], ls_p4.x[1], ls_p4.x[2]]

    ls_p5 = np.mean(x)
    dictplanep['paralleltoyz'] = [1, 0, 0, -ls_p5]

    ls_p6 = np.mean(y)
    dictplanep['paralleltoxz'] = [0, 1, 0, -ls_p6]

    ls_p7 = np.mean(z)
    dictplanep['paralleltoxy'] = [0, 0, 1, -ls_p7]

    minesum = 100
    minkey = None
    for key in dictplanep.keys():
        dictplanep[key] = [round(i, 6) for i in dictplanep[key]]
        if np.sum(np.array(dictplanep[key])[:-1] ** 2) > 1e-5:
            x_project, y_project, z_project = \
                projected_on_plane(dictplanep[key], (x, y, z))
            # disto = np.max((x_project-x)**2+(y_project-y)**2+(z_project-z)**2)
            esum = np.sum((x_project - x) ** 2 
                          + (y_project - y) ** 2 + (z_project - z) ** 2)
            if minesum > esum:
                minesum = esum
                minkey = key

    return dictplanep[minkey]


def point_on_plane(params, x, y):
    aa, bb, cc, dd = params
    if abs(cc) > 1e-7:
        if abs(aa) > 1e-7 or abs(bb) > 1e-7:
            z = (-aa * x - bb * y - dd) / cc
        else:
            z = -np.ones_like(x) * dd / cc
    else:
        z = np.random.uniform(-x.max(), x.max(), x.shape)
        if abs(aa) > 1e-7 and abs(bb) > 1e-7:
            y = (-aa * x - dd) / bb
        elif abs(aa) < 1e-7:
            y = np.ones_like(x) * dd / bb
        elif abs(bb) < 1e-7:
            x = np.ones_like(x) * dd / aa
    return x, y, z


def projected_on_plane(params, points):
    # aa*x + bb*y + cc*z + dd = 0

    aa, bb, cc, dd = params
    # if len(params)==4:
    #     (aa, bb, cc, dd) = params
    # elif len(params)==3:
    #     (aa, bb, dd) = params
    # cc = -1

    (xo, yo, zo) = points

    xp = (((bb ** 2 + cc ** 2) * xo 
           - aa * (bb * yo + cc * zo + dd)) / (aa ** 2 + bb ** 2 + cc ** 2))
    yp = (((aa ** 2 + cc ** 2) * yo 
           - bb * (aa * xo + cc * zo + dd)) / (aa ** 2 + bb ** 2 + cc ** 2))
    zp = (((aa ** 2 + bb ** 2) * zo 
           - cc * (aa * xo + bb * yo + dd)) / (aa ** 2 + bb ** 2 + cc ** 2))

    return xp, yp, zp


def longest_vector(params, x, y, z):
    x_project, y_project, z_project = projected_on_plane(params, (x, y, z))

    points = (np.vstack([x_project, y_project, z_project]).T)
    # obtain the pairwise distances
    distance = squareform(pdist(points))
    # get the max distance
    maxidx = np.argmax(distance)
    # get the 2-D index i.e. the two corresponding points
    idx1, idx2 = divmod(maxidx, distance.shape[1])
    point1 = np.array([x_project[idx1], y_project[idx1], z_project[idx1]])
    point2 = np.array([x_project[idx2], y_project[idx2], z_project[idx2]])

    lvector = point2 - point1
    if lvector[0] < 0:  # make sure the rotate angle relative to x-axis is positive
        lvector = -lvector

    return lvector, point1, point2


def fit_ellipse(a, c, u, v, ct, st, points):
    cx, cy, cz = c
    xp, yp, zp = points

    eq = ((cx + a[0] * u[0] * ct + a[1] * v[0] * st - xp) ** 2 +
          (cy + a[0] * u[1] * ct + a[1] * v[1] * st - yp) ** 2 +
          (cz + a[0] * u[2] * ct + a[1] * v[2] * st - zp) ** 2).mean()
    return eq


def get_elliptic_cylinder(a, b, u, v, n, center):
    [cx, cy, cz] = center
    theta = np.linspace(0, np.pi * 2, 100)
    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    rotate = np.array([[u[0], u[1]],
                       [-u[1], u[0]]])

    new = np.dot(rotate, np.vstack([costheta, sintheta]))

    costheta2 = new[0, :]
    sintheta2 = new[1, :]

    xx = np.zeros((50, 100))
    yy = np.zeros((50, 100))
    zz = np.zeros((50, 100))
    tt = np.linspace(-2, 2, 50)
    for ii in range(50):
        xx[ii, :] = (
            cx + n[0] * tt[ii] + a * u[0] * costheta2 + b * v[0] * sintheta2)
        yy[ii, :] = (
            cy + n[1] * tt[ii] + a * u[1] * costheta2 + b * v[1] * sintheta2)
        zz[ii, :] = (
            cz + n[2] * tt[ii] + a * u[2] * costheta2 + b * v[2] * sintheta2)

    return xx, yy, zz


def getting_fitted_ellipses(points, requirements):
    x, y, z = points

    rings = []
    pcurves = []
    rotate_theta_list = []
    pns = []
    r2_pc_list = []
    r2_pl_list = []

    # fit plane
    plane_params = fit_planes(x, y, z)
    if plane_params[2] < 0:
        plane_params = [-i for i in plane_params]
    # project point onto plane

    x_project, y_project, z_project = projected_on_plane(plane_params, 
                                                         (x, y, z))

    x_plane = np.linspace(-1, 1, 100)
    y_plane = np.linspace(-1, 1, 100)
    x_plane, y_plane = np.meshgrid(x_plane, y_plane)
    x_plane, y_plane, z_plane = point_on_plane(plane_params, x_plane, y_plane)

    if requirements == 'plane':
        return (x_plane, y_plane, z_plane), \
            (x_project, y_project, z_project), plane_params

    # get normal vector
    p_n = np.array([plane_params[0], plane_params[1], plane_params[2]])
    p_n = p_n / np.linalg.norm(p_n)
    pns.append(p_n)

    # get long axis vector
    p_u, p1, p2 = longest_vector(plane_params, x, y, z)
    p_u = p_u / np.linalg.norm(p_u)
    # p_u = np.array([1, 0, point_on_plane(plane_params, 1, 0)]) - np.array([0, 0, point_on_plane(plane_params, 0, 0)])
    # p_u = p_u/np.linalg.norm(p_u)
    assert (np.dot(p_n, p_u) < 1e-10)

    # get short axis vector
    p_v = np.cross(p_n, p_u)
    if p_v[1] < 0:
        p_v = -p_v
    p_v = p_v / np.linalg.norm(p_v)

    # get center
    cx = x_project.min() + np.ptp(x_project) / 2
    cy = y_project.min() + np.ptp(y_project) / 2  # np.mean(x_project), np.mean(y_project)
    cx, cy, cz = point_on_plane(plane_params, cx, cy)  # plane_params[0]*cx + plane_params[1]*cy + plane_params[2]

    # get theta of points
    costheta = x_project / np.sqrt(x_project ** 2 + y_project ** 2)
    sintheta = y_project / np.sqrt(x_project ** 2 + y_project ** 2)

    # rotate matrix to align x-axis to p_u
    rotate = np.array([[p_u[0], p_u[1]],
                       [-p_u[1], p_u[0]]])

    rotate_theta = np.arctan2(-p_u[1], p_u[0])
    rotate_theta_list.append(rotate_theta)

    new = np.dot(rotate, np.vstack([costheta, sintheta]))

    costheta20 = new[0, :]
    sintheta20 = new[1, :]

    el = optimize.least_squares(fit_ellipse, [1, 1],
                                args=((cx, cy, cz), p_u, p_v,
                                      costheta20, sintheta20,
                                      (x_project, y_project, z_project)))
    # bounds=(0, axises*1.2))

    a, b = el.x

    # ellipse-fitted point
    xxp = cx + a * p_u[0] * costheta20 + b * p_v[0] * sintheta20
    yyp = cy + a * p_u[1] * costheta20 + b * p_v[1] * sintheta20
    zzp = cz + a * p_u[2] * costheta20 + b * p_v[2] * sintheta20

    ## for continuous curves
    theta = np.linspace(0, np.pi * 2, 100)

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    rotate = np.array([[p_u[0], p_u[1]],
                       [-p_u[1], p_u[0]]])

    new = np.dot(rotate, np.vstack([costheta, sintheta]))

    costheta2 = new[0, :]
    sintheta2 = new[1, :]

    xxpp = cx + a * p_u[0] * costheta2 + b * p_v[0] * sintheta2
    yypp = cy + a * p_u[1] * costheta2 + b * p_v[1] * sintheta2
    zzpp = cz + a * p_u[2] * costheta2 + b * p_v[2] * sintheta2

    if requirements == 'ellipses':
        return (xxpp, yypp, zzpp), (xxp, yyp, zzp), (a, b, p_u, p_v, (cx, cy, cz))


def r_square(yfit, yactual):
    # yfit and yactual have same shape as (3 x nTrial)
    sse = np.sum((yactual - yfit) ** 2, axis=1)
    sst = np.sum((yactual - np.tile(np.mean(yactual, axis=1)[:, np.newaxis],
                                    (1, yactual.shape[1]))) ** 2, axis=1)

    return 1 - sse / sst


# @title @getting_angle_between_planes

def getting_angle_between_planes(pcs, SPIdx, SPList):
    norvs_list = []
    for isp in range(len(np.unique(SPIdx))):
        pcs_points = (pcs[SPIdx == isp, 0], 
                      pcs[SPIdx == isp, 1], 
                      pcs[SPIdx == isp, 2])

        plane_points, pp_points, pparams = getting_fitted_ellipses(pcs_points, 
                                                                   'plane')
        # plot_fitted_plane(pcs_points, plane_points)

        param_a, param_b, param_c, param_d = pparams
        norv = np.array([param_a, param_b, param_c])
        norv = norv / np.linalg.norm(norv)
        norvs_list.append(norv)

    theta_pp_list = []
    co_idx = np.argwhere(SPList == 0)[0, 0]
    for isp in range(len(np.unique(SPIdx))):
        theta_pp = np.arccos(np.dot(norvs_list[isp],
                                    norvs_list[co_idx]))
        if np.cross(norvs_list[isp], norvs_list[co_idx])[-1] < 0:
            theta_pp = -theta_pp
        theta_pp_list.append(theta_pp)

    theta_pp_np = np.array(theta_pp_list)
    theta_pp_np = theta_pp_np / np.pi * 180

    theta_xy_list = []
    theta_z_list = []
    for isp in range(len(np.unique(SPIdx))):
        theta_xy = np.arctan2(norvs_list[isp][1], norvs_list[isp][0])
        theta_xy_list.append(theta_xy)

        theta_z = np.arctan2(norvs_list[isp][2],
                             np.linalg.norm(norvs_list[isp][:2]))
        theta_z_list.append(theta_z)

    theta_xy_np = np.array(theta_xy_list) / np.pi * 180
    theta_z_np = np.array(theta_z_list) / np.pi * 180

    return theta_pp_np, theta_xy_np, theta_z_np


# @title @getting_fitted_pancake_curves_standard

def getting_fitted_pancake_curves_standard(pcs_points):
    x, y, z = pcs_points

    # fit plane
    plane_params = fit_planes(x, y, z)
    if plane_params[2] < 0:
        plane_params = [-i for i in plane_params]

    # get normal vector
    p_n = np.array([plane_params[0], plane_params[1], plane_params[2]])
    p_n = p_n / np.linalg.norm(p_n)

    # get long axis vector
    p_u, p1, p2 = longest_vector(plane_params, x, y, z)
    p_u = p_u / np.linalg.norm(p_u)
    # p_u = np.array([1, 0, point_on_plane(plane_params, 1, 0)]) - np.array([0, 0, point_on_plane(plane_params, 0, 0)])
    # p_u = p_u/np.linalg.norm(p_u)
    assert (np.dot(p_n, p_u) < 1e-10)

    # get short axis vector
    p_v = np.cross(p_n, p_u)
    if p_v[1] < 0:
        p_v = -p_v
    p_v = p_v / np.linalg.norm(p_v)

    # get center
    x_project, y_project, z_project = projected_on_plane(plane_params, 
                                                         (x, y, z))
    cx = x_project.min() + np.ptp(x_project) / 2
    cy = y_project.min() + np.ptp(y_project) / 2  # np.mean(x_project), np.mean(y_project)
    cx, cy, cz = point_on_plane(plane_params, cx, cy)  # plane_params[0]*cx + plane_params[1]*cy + plane_params[2]

    # get theta of points
    costheta = x_project / np.sqrt(x_project ** 2 + y_project ** 2)
    sintheta = y_project / np.sqrt(x_project ** 2 + y_project ** 2)

    # rotate matrix to align x-axis to p_u
    rotate = np.array([[p_u[0], p_u[1]],
                       [-p_u[1], p_u[0]]])

    rotate_theta = np.arctan2(-p_u[1], p_u[0])

    new = np.dot(rotate, np.vstack([costheta, sintheta]))

    costheta20 = new[0, :]
    sintheta20 = new[1, :]

    el = optimize.least_squares(fit_ellipse, [1, 1],
                                args=((cx, cy, cz), p_u, p_v,
                                      costheta20, sintheta20,
                                      (x_project, y_project, z_project)))
    # bounds=(0, axises*1.2))

    a, b = el.x

    # ellipse-fitted point
    xxp = cx + a * p_u[0] * costheta20 + b * p_v[0] * sintheta20
    yyp = cy + a * p_u[1] * costheta20 + b * p_v[1] * sintheta20
    zzp = cz + a * p_u[2] * costheta20 + b * p_v[2] * sintheta20

    def fit_hyperbolic_paraboloid(p, x, y, z):
        c, d, beta = p
        xxhp = (xxp + c * p_n[0] * (2 * (costheta20 * np.cos(beta)
                                         + sintheta20 * np.sin(beta)) ** 2 - 1) 
                + d * p_n[0])
        yyhp = (yyp + c * p_n[1] * (2 * (costheta20 * np.cos(beta) 
                                         + sintheta20 * np.sin(beta)) ** 2 - 1) 
                + d * p_n[1])
        zzhp = (zzp + c * p_n[2] * (2 * (costheta20 * np.cos(beta) 
                                         + sintheta20 * np.sin(beta)) ** 2 - 1) 
                + d * p_n[2])

        eq = np.sum((x - xxhp) ** 2 + (y - yyhp) ** 2 + (z - zzhp) ** 2)
        return eq

    # fit hyperbolic paraboloid
    ls_hp = optimize.least_squares(fit_hyperbolic_paraboloid, np.ones((3,)),
                                   args=(x, y, z))

    c, d, beta = ls_hp.x
    xxhp = (xxp + c * p_n[0] * (2 * (costheta20 * np.cos(beta) 
                                     + sintheta20 * np.sin(beta)) ** 2 - 1) 
            + d * p_n[0])
    yyhp = (yyp + c * p_n[1] * (2 * (costheta20 * np.cos(beta) 
                                     + sintheta20 * np.sin(beta)) ** 2 - 1) 
            + d * p_n[1])
    zzhp = (zzp + c * p_n[2] * (2 * (costheta20 * np.cos(beta) 
                                     + sintheta20 * np.sin(beta)) ** 2 - 1) 
            + d * p_n[2])

    ## for continuous curves
    theta = np.linspace(0, np.pi * 2, 100)

    costheta = np.cos(theta)
    sintheta = np.sin(theta)

    rotate = np.array([[p_u[0], p_u[1]],
                       [-p_u[1], p_u[0]]])

    new = np.dot(rotate, np.vstack([costheta, sintheta]))

    costheta2 = new[0, :]
    sintheta2 = new[1, :]

    xxhpp = (cx 
             + a * p_u[0] * costheta2 + b * p_v[0] * sintheta2
             + c * p_n[0] * (2 * (costheta2 * np.cos(beta) 
                                  + sintheta2 * np.sin(beta)) ** 2 - 1) 
             + d * p_n[0])
    yyhpp = (cy 
             + a * p_u[1] * costheta2 + b * p_v[1] * sintheta2
             + c * p_n[1] * (2 * (costheta2 * np.cos(beta) 
                                  + sintheta2 * np.sin(beta)) ** 2 - 1) 
             + d * p_n[1])
    zzhpp = (cz 
             + a * p_u[2] * costheta2 + b * p_v[2] * sintheta2
             + c * p_n[2] * (2 * (costheta2 * np.cos(beta) 
                                  + sintheta2 * np.sin(beta)) ** 2 - 1) 
             + d * p_n[2])

    return (xxhpp, yyhpp, zzhpp), (xxhp, yyhp, zzhp), (a, b, p_u, p_v, (cx, cy, cz), c, d, beta)
