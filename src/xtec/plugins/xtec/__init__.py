from . import cluster_data


def plugin_menu():
    menu = "XTEC"
    actions = [(("Cluster Data", cluster_data.show_dialog))]

    return menu, actions
