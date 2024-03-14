import numpy as np
import itertools
from collections import defaultdict
import re


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def find_matching_index(list1, list2):
    inverse_index = defaultdict(list)
    for index, element in enumerate(list1):
        inverse_index[element].append(index)
    matching_index = [inverse_index[element] for element in set(list2) if element in inverse_index]
    return list(itertools.chain.from_iterable(matching_index))


def find_string_match(input, string):
    sta = re.search(input, string).start()
    end = re.search(input, string).end()
    return sta, end
