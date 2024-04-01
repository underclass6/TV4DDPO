#########################################
# temperature schedulers
#########################################


def step_scheduler(t):
    # time stamps
    t1 = 5
    t2 = 10
    t3 = 15
    # temperature stamps
    tem_max = 10
    tem_min = 0.1
    tem1 = 2
    tem2 = 1

    if t < t1:
        return tem_max
    if t1 <= t < t2:
        return tem1
    if t2 <= t < t3:
        return tem2
    if t3 <= t:
        return tem_min


def linear_scheduler(t):
    t_max = 100
    tem_max = 100
    tem_min = 1

    return (tem_min - tem_max) / t_max * t + tem_max


def exponential_scheduler(t):
    beta = 0.5
    tem_max = 100

    return beta ** t * tem_max
