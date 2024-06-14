def check_layer(layer, p):
    # if p is tuple, check layer validity
    if isinstance(p, tuple):
        p, sampled_layer = p
        assert layer == sampled_layer, \
            "Layer mismatch, check order of layers " \
            "in your model implementation. " \
            f"Expected {sampled_layer} but got {layer}"
    return p


def group_lasso_criterion(model, hierarchical=False):
    group_lasso_loss = 0
    for m in model.modules():
        if hasattr(m, 'inner_dim'):
            gl = m.group_lasso(hierarchical=hierarchical)
            group_lasso_loss += gl.sum()
    return group_lasso_loss


def progressive_shrinking(model, threshold, hierarchical=False):
    # progressive shrinking
    for m in model.modules():
        if hasattr(m, 'inner_dim'):
            m.assign_low_rank(threshold, hierarchical=hierarchical)


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # use cumulative moving average
        m.momentum = None
