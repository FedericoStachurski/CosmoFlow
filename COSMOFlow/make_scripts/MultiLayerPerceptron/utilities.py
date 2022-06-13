import os

def get_script_path():
    #return os.path.dirname(os.path.realpath(sys.argv[0]))
    return os.getcwd()


def norm(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if ref_dataframe is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_dataframe.mean())/ref_dataframe.std()
        elif norm_type == 'uniform':
            df_norm = 2*((dataframe - ref_dataframe.min())/ref_dataframe.max()) - 1
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_inputs[0])/ref_inputs[1]
        elif norm_type == 'uniform':
            df_norm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
        else:
            raise InputError("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_norm


def norm_inputs(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if ref_dataframe is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_dataframe.mean(axis=0))/ref_dataframe.std(axis=0)
        elif norm_type == 'uniform':
            df_norm = 2*((dataframe - ref_dataframe.min(axis=0))/ref_dataframe.max(axis=0)) - 1
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_norm = (dataframe - ref_inputs[0])/ref_inputs[1]
        elif norm_type == 'uniform':
            df_norm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
        else:
            raise InputError("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_norm


#def norm_inputs(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
#    if ref_dataframe is not None:
#        df_norm = (dataframe - ref_dataframe.mean(axis=0)) / ref_dataframe.std(axis=0)
#    elif ref_mean is not None and ref_std is not None:
#        df_norm = (dataframe - ref_mean) / ref_std
#    else:
#        raise RuntimeError("Either a reference dataset or a reference mean + std must be supplied.")
#    return df_norm


def unnorm(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if ref_dataframe is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_dataframe.std()) + ref_dataframe.mean()
        elif norm_type == 'uniform':
            #df_unnorm = 2*((dataframe - np.min(ref_dataframe))/np.max(ref_dataframe)) - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_dataframe.max() + ref_dataframe.min()
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_inputs[1]) + ref_inputs[0]
        elif norm_type == 'uniform':
            #df_unnorm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_inputs[1] + ref_inputs[0]
        else:
            raise InputError("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_unnorm


#def unnorm(dataframe_norm, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
#    if ref_dataframe is not None:
#        dataframe = (dataframe_norm * ref_dataframe.std()) + ref_dataframe.mean()
#    elif ref_
#        dataframe = (dataframe_norm * ref_std) + ref_mean
#    else:
#        raise RuntimeError("Either a reference dataset or a reference mean + std must be supplied.")
#    return dataframe

def unnorm_inputs(dataframe, ref_dataframe=None, ref_inputs=None, norm_type='z-score'):
    if ref_dataframe is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_dataframe.std(axis=0)) + ref_dataframe.mean(axis=0)
        elif norm_type == 'uniform':
            #df_unnorm = 2*((dataframe - np.min(ref_dataframe))/np.max(ref_dataframe)) - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_dataframe.max(axis=0) + ref_dataframe.min(axis=0)
    elif ref_inputs is not None:
        if norm_type == 'z-score':
            df_unnorm = (dataframe * ref_inputs[1]) + ref_inputs[0]
        elif norm_type == 'uniform':
            #df_unnorm = 2*(dataframe - ref_inputs[0])/ref_inputs[1] - 1
            df_unnorm = 0.5*(dataframe + 1) * ref_inputs[1] + ref_inputs[0]
        else:
            raise InputError("normalisation must be z-score or uniform")
    else:
        raise RuntimeError("Either a reference dataset or a set of reference inputs must be supplied")
    return df_unnorm


#def unnorm_inputs(dataframe_norm, ref_dataframe,ref_mean=None, ref_std=None, norm_type='z-score'):
#    if ref_dataframe is not None:
#        dataframe = (dataframe_norm * ref_dataframe.std(axis=0)) + ref_dataframe.mean(axis=0)
#    elif ref_mean is not None and ref_std is not None:
#        dataframe = (dataframe_norm * ref_std) + ref_mean
#    else:
#        raise RuntimeError("Either a reference dataset or a reference mean + std must be supplied.")
#    return dataframe
