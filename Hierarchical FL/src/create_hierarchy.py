
import numpy as np
import copy
from options import args_parser
from utils import average_weights
from update import test_inference
import torch


class Structure(object):
    """ create hierarchical structure, help user download weight, update weights
     on server"""
    def __init__(self, args, global_weights, global_model, test_dataset=None):
        self.nn_architecture = global_model
        self.nn_weights = global_weights
        self.args = args
        self.system = self.hierarchy()
        self.layers = len(args.mid_server) + 1
        self.dataset = test_dataset

    # To do: when there is no middle layers
    def hierarchy(self):
        """
        the hierarchical structure is represented as follows
        { -1: {user_idx: local weight} # user layer
          0: {server_idx: [user_set, aggregated weight]} # edge server layer
          1: {sever_idx: [server_set, aggregated weight]} # higher server layer
          ...
        }
        """
        structure = {}

        def build(layer_idx):

            layer_dict = {}

            # the user layer: user_idx: weights
            if layer_idx == -1:
                for user in range(self.args.num_users):
                    layer_dict[user] = self.nn_weights

            # the last layer is the central server
            if layer_idx == len(self.args.mid_server):
                server_set = [i for i in range(self.args.mid_server[layer_idx-1])]
                layer_dict[0] = [server_set, self.nn_weights]
                structure[layer_idx] = layer_dict
                return

            num_server = self.args.mid_server[layer_idx]

            # the first layer is connected with users
            if layer_idx == 0:
                all_user = [i for i in range(self.args.num_users)]
                user_per_server = self.args.num_users // num_server

                # create the user set
                for server in range(num_server):
                    if server == (num_server-1):
                        user_set = set(all_user)
                    else:
                        user_set = set(np.random.choice(all_user, user_per_server,
                                                 replace=False))
                    layer_dict[server] = [user_set, self.nn_weights]
                    all_user = list(set(all_user) - user_set)

            # other mid_layer
            elif layer_idx > 0:
                num_prev_server = self.args.mid_server[layer_idx - 1]
                server_per_server = num_prev_server // num_server
                all_server = [i for i in range(num_prev_server)]

                for server in range(num_server):
                    if server == (num_server - 1):
                        server_set = set(all_server)
                    else:
                        server_set = set(np.random.choice(all_server, server_per_server,
                                                 replace=False))

                    layer_dict[server] = [server_set, self.nn_weights]
                    all_server = list(set(all_server) - server_set)

            structure[layer_idx] = layer_dict
            build(layer_idx + 1)

        build(-1)

        return structure

    # download model from server or use local model
    def get_model(self, user_idx, download):
        """
        download model for the user: user_idx or use local model on the user
        """
        first_layer_server = self.system[0]
        model = copy.deepcopy(self.nn_architecture)

        for server_idx in first_layer_server.keys():
            if user_idx in first_layer_server[server_idx][0]:

                if download:
                    model_weights = copy.deepcopy(first_layer_server[server_idx][1])
                else:
                    model_weights = copy.deepcopy(self.system[-1][user_idx])

                model.load_state_dict(model_weights)
                return model, server_idx

    # To do: how can we do model management? Do we need to test on all models?
    def upload_weights(self, local_weights):
        """
        upload updated weights after local training on users to servers
        and do model aggregation on all servers
        """

        def update(layer_idx):
            if layer_idx == self.layers:
                return

            if layer_idx == 0:
                for server_idx in local_weights.keys():
                    self.system[layer_idx][server_idx][1] = average_weights(
                            local_weights[server_idx])

            else:
                for server_idx in self.system[layer_idx]:
                    prev_server_weights = []
                    for prev_server_idx in self.system[layer_idx][server_idx][0]:
                        prev_server_weights.append(self.system[layer_idx-1]
                                                   [prev_server_idx][1])

                    self.system[layer_idx][server_idx][1] = average_weights(
                        prev_server_weights)

            update(layer_idx+1)

        # weights on user have already been updated so started from the first layer of server
        update(0)

    # top-down updates ensure the model on the bottom-layer servers is the best
    def model_management(self):
        """
        top-down model management (bubble sort) to ensure the best model
        is on the edge and available to users
        """

        def compare_accuracy(w1, w2):
            model = self.nn_architecture
            model.load_state_dict(w1)
            acc1, loss1 = test_inference(self.args, model, self.dataset)
            model.load_state_dict(w2)
            acc2, loss2 = test_inference(self.args, model, self.dataset)
            return True if acc1 >= acc2 else False

        def manage(layer_idx):
            if layer_idx == 0:
                return

            for server in self.system[layer_idx]:
                w1 = self.system[layer_idx][server][1]
                for server_idx in self.system[layer_idx][server][0]:
                    w2 = self.system[layer_idx-1][server_idx][1]
                    if compare_accuracy(w1, w2):
                        w2 = copy.deepcopy(w1)

            manage(layer_idx-1)

        # top-down model management
        manage(self.layers-1)

    def save_model(self):
        """
        user model name: connecting server + server_idx + user + user_idx
        server_idx is the server that the user connect to

        server model name: layer + layer_idx + server + server + idx
        """

        for layer_idx in range(-1, self.layers):
            if layer_idx == -1:
                for user_idx in self.system[layer_idx]:
                    # find the server_idx
                    for server_idx in self.system[0]:
                        if user_idx in self.system[0][server_idx][0]:
                            break

                    model_path = '../save/trained_models/' + 'connecting_server' + str(server_idx) + 'user' + \
                        str(user_idx) + '.pth'
                    torch.save(self.system[layer_idx][user_idx], model_path)

            else:
                for server_idx in self.system[layer_idx]:
                    model_path = '../save/trained_models/' + 'layer' + str(layer_idx) + 'server' + \
                                 str(server_idx) + '.pth'
                    torch.save(self.system[layer_idx][server_idx][1], model_path)


# show the hierarchical structure
if __name__ == '__main__':
    args = args_parser()
    structure = Structure(args, 1, 2)
    print(structure.system)
