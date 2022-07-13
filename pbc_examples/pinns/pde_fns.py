from pbc_examples.pinns.gradients import jacobian, hessian

time_pdes = ['advection', 'allen_cahn']

def advection(x, y):
    # j: input, i: output
    velocity = 3
    dy_t = jacobian(y, x, i=0, j=1)
    dy_x = jacobian(y, x, i=0, j=0)
    return dy_t + velocity * dy_x


def allen_cahn(x, y):
    d = 0.001
    dy_t = jacobian(y, x, i=0, j=1)
    dy_xx = hessian(y, x, i=0, j=0)
    return dy_t - d * dy_xx - 5 * (y - y ** 3)
