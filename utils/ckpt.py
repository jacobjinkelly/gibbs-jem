"""
For saving and loading checkpoints.
"""

import os
import time
from operator import itemgetter

import torch

_WITH_STATE = ["models", "optimizer"]


def save_ckpt(itr, logger, device, data, save_dir=None, ckpt_path=None, overwrite=True,
              most_recent=None, save_best=None, tst_accs=None, prefix=""):
    """
    Save a checkpoint in case job is prempted.
    """

    ckpt_file_ext = ".pt"

    assert (most_recent is None) or (save_best is None)

    def get_path_from_itr(itr_):
        return os.path.join(save_dir, f"{prefix}_{itr_:06d}{ckpt_file_ext}")

    def get_itr_from_path(path_):
        path_ = path_[len(prefix):-len(ckpt_file_ext)]  # remove prefix and file extension
        assert path_[0] == "_", f"Could not parse format of {path_}"
        return int(path_[1:])  # remove underscore and cast to int

    def get_itrs_from_dir():
        return [get_itr_from_path(f) for f in os.listdir(save_dir)
                if f.startswith(prefix) and f.endswith(ckpt_file_ext)]

    if most_recent is not None:
        if not isinstance(most_recent, int):
            raise ValueError(f"Got {most_recent} of type {type(most_recent)} instead of type int for most_recent")

        if save_dir is None:
            raise ValueError(f"Save dir must not be none")

        if prefix == "":
            raise ValueError(f"Prefix must be non-empty, got {prefix}")

        path = get_path_from_itr(itr)
    elif save_best is not None:
        if not isinstance(save_best, int):
            raise ValueError(f"Got {save_best} of type {type(save_best)} instead of type int for most_recent")

        if save_dir is None:
            raise ValueError(f"Save dir must not be none")

        if prefix == "":
            raise ValueError(f"Prefix must be non-empty, got {prefix}")

        if tst_accs is None:
            raise ValueError(f"Tst accs must not be None when saving best weights.")

        path = get_path_from_itr(itr)
    elif overwrite and prefix == "":
        # overwrite the same checkpoint since it's just used for preemption
        assert isinstance(ckpt_path, str)
        path = ckpt_path
    else:
        assert isinstance(save_dir, str)
        if overwrite and prefix != "":
            path = os.path.join(save_dir, f"{prefix}{ckpt_file_ext}")
        else:
            path = get_path_from_itr(itr)

    try:
        for model in data["models"]:
            data["models"][model].cpu()

        ckpt_data = {}
        for datum in data:
            if datum in _WITH_STATE:
                ckpt_data[datum] = {}
                for obj in data[datum]:
                    ckpt_data[datum][obj] = data[datum][obj].state_dict()
            else:
                ckpt_data[datum] = data[datum]

        save_new_best_model = (save_best is not None and itr in get_top_k_accs(tst_accs, save_best))
        if save_best is None or save_new_best_model:
            torch.save(ckpt_data, path)
        
        for model in data["models"]:
            data["models"][model].to(device)

        # remove the old checkpoint if it exists
        if most_recent is not None:
            start_find = time.time()
            recent_itrs = get_itrs_from_dir()
            time_find = time.time() - start_find

            assert len(set(recent_itrs)) == len(recent_itrs), f"Found duplicate itrs in {recent_itrs} for most_recent"
            assert all(0 < itr_ <= itr for itr_ in recent_itrs), f"Found invalid itr in {recent_itrs} for most_recent"

            logger(f"Took {time_find:.4e} s to find most_recent checkpoints.")

            if len(recent_itrs) == most_recent + 1:
                start_remove = time.time()

                # find the smallest one and remove it
                itr_to_remove = min(recent_itrs)
                path_to_remove = get_path_from_itr(itr_to_remove)

                # remove the path
                os.remove(path_to_remove)

                time_remove = time.time() - start_remove

                logger(f"Removed {path_to_remove} (took {time_remove:.4e} s).")

            else:
                if len(recent_itrs) > most_recent:
                    logger(f"Found more checkpoints than there should be for most_recent")

                while len(recent_itrs) > most_recent:
                    start_remove = time.time()

                    # find the smallest one remove it
                    itr_to_remove = min(recent_itrs)
                    path_to_remove = get_path_from_itr(itr_to_remove)

                    # remove the path
                    os.remove(path_to_remove)
                    recent_itrs.remove(itr_to_remove)

                    time_remove = time.time() - start_remove

                    logger(f"Removed {path_to_remove} (took {time_remove:.4e} s).")
        elif save_new_best_model:
            start_find = time.time()
            best_itrs = get_itrs_from_dir()
            time_find = time.time() - start_find

            assert len(set(best_itrs)) == len(best_itrs), f"Found duplicate itrs in {best_itrs} for save_best"
            assert all(0 < itr_ <= itr for itr_ in best_itrs), f"Found invalid itr in {best_itrs} for save_best"

            logger(f"Took {time_find:.4e} s to find save_best checkpoints.")

            start_remove = time.time()

            if len(best_itrs) == save_best + 1:
                # find the itr with the least accuracy and remove it
                itr_to_remove = min([(best_itr, tst_accs[best_itr]) for best_itr in best_itrs], key=itemgetter(1))[0]
                path_to_remove = get_path_from_itr(itr_to_remove)

                # remove the path
                os.remove(path_to_remove)

                time_remove = time.time() - start_remove

                logger(f"Removed {path_to_remove} (took {time_remove:.4e} s).")
            else:
                assert len(best_itrs) <= save_best

    except IOError:
        logger(f"Unable to save {path} at iteration {itr}")


def get_top_k_accs(tst_accs, k):
    """
    :param tst_accs: Dictionary mapping itr to accuracy.
    :param k: Number of top accuracies to compute.
    :return: Itrs of top k accuracies.
    """
    return list(map(itemgetter(0), sorted(tst_accs.items(), key=itemgetter(1), reverse=True)))[:k]
