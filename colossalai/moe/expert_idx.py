from multiprocessing import Manager

manager = Manager()
expert_idxs  = manager.list()
# expert_idxs = [0, 1, 2]

def set_expert_idxs(idxs):
    assert isinstance(idxs, list)
    expert_idxs = idxs
    print("fixed expert idxs to {}".format(expert_idxs))

def get_expert_idxs():
    return expert_idxs