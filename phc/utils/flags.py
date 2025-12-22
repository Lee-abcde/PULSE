__all__ = ['flags', 'summation']

class Flags(object):
    def __init__(self, items):
        for key, val in items.items():
            setattr(self,key,val)

flags = Flags({
    'test': False, 
    'debug': False,
    "real_traj": False,
    "im_eval": False,

    "freq_inc": False,
    "freq_dec": False,
    "P_lower_dec": False,
    "P_lower_inc": False,
    "P_upper_dec": False,
    "P_upper_inc": False,
    "Index_dec": False,
    "Index_inc": False,
    "reset": False,
    "text_change": False
    })
