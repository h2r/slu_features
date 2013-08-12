threshold = 0.0000001
import math
import numpy as na



def sorta_eq(v1, v2, threshold=threshold):
    return math.fabs(v1 - v2) < threshold
def array_equal(arr1, arr2):
    arr1 = na.array(arr1)
    arr2 = na.array(arr2)
    if arr1 == None or arr2 == None:
        return False
    elif len(arr1) != len(arr2):
        return False
    elif arr1.shape != arr2.shape:
        return False
    else:
        return (na.fabs(na.array(arr1) - na.array(arr2)) < threshold).all()
def assert_sorta_eq(v1, v2):
    assert sorta_eq(v1, v2), (v1, v2)

def assert_array_equal(arr1, arr2):
    try:
        for a1, a2 in zip(arr1, arr2):
            if hasattr(a1, "__iter__") and hasattr(a2, "__iter__"):
                assert_array_equal(a1, a2)
            else:
                assert_sorta_eq(a1, a2)            
        assert len(arr1) == len(arr2), (arr1, arr2)
    except:
        print "arr1", arr1
        print "arr2", arr2
        raise
    #assert array_equal(arr1, arr2), (arr1, arr2)



def has_unique_maximum(lst):
    """
    Whether the list has a unique maximum.
    """
    if len(lst) == 0:
        return False
    max_idx = na.argmax(lst)
    max_val = lst[max_idx]
    
    values = [v for v in lst if v == max_val]
    if len(values) == 1:
        return True
    elif len(values) > 1:
        return False
    else:
        raise ValueError("Should never get here: " + `values`)


def assert_sparse_dict_equal(d1, d2):
    for key in sorted(set(d1.keys() + d2.keys())):
        try:
            value1 = d1.get(key, 0.0)
            value2 = d2.get(key, 0.0)
            assert value1 == value2, (key, value1, value2)
        except KeyError:
            print "key", key
            print "in d1", key in d1
            print "in d2", key in d2
            raise

    
aeq = assert_sorta_eq
