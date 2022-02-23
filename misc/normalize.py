'''
################################
#        L2 NORMALIZATION      #
################################
'''

def l2_norm(x):


    norm = x.pow(2).sum().pow(1. /2)
    out = x.div(norm)

    return out
