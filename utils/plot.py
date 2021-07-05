"""
Plot model samples.
"""
from functools import partial

from .utils import wrap_args
from .data import ALL_TOY_DSETS


@wrap_args
def plot_model_samples(plot, model, itr,
                       x_train, fixed_a_train, fixed_c_train,
                       mode, p_x_weight, p_x_given_y_weight, p_xy_weight, dataset,
                       **_):
    if dataset in ALL_TOY_DSETS:
        plot = partial(plot, model)
    plot(x_train, "data", itr)
    if mode == "ebm":
        plot(model.sample(fixed_noise=True), "uncond_fixed", itr)
        plot(model.sample(), "uncond", itr)
    elif mode == "poj":
        if p_x_weight > 0:
            plot(model.sample_logp_x(fixed_noise=True), "uncond_fixed", itr)
            plot(model.sample_logp_x(), "uncond", itr)

        if p_x_given_y_weight > 0:
            plot(model.sample_logp_x_given_y(fixed_a_train, fixed_c_train, fixed_noise=True),
                 "cond_fixed", itr)
            plot(model.sample_logp_x_given_y(fixed_a_train, fixed_c_train),
                 "cond", itr)

        if p_xy_weight > 0:
            plot(model.sample_logp_xy(fixed_noise=True),
                 "joint_fixed", itr)
            plot(model.sample_logp_xy(),
                 "joint", itr)
