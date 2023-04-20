from . import GMM_clustering, BIC_score


def plugin_menu():
    menu = "XTEC"
    actions = []
    actions.append(("XTEC", GMM_clustering.show_dialog))
    actions.append(("BIC score", BIC_score.show_dialog))

    return menu, actions
