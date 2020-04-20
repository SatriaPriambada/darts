import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from operations import *
from genotypes import PRIMITIVES
from genotypes import Genotype
from model_search import Network
from model_search import MixedOp
from model_search import Cell
from model import HeterogenousNetworkImageNet
from model import HeterogenousNetworkCIFAR
import random
import itertools
import re
import mcts
import latency_profiler


class MacroNetwork(nn.Module):
    def __init__(
        self,
        C,
        num_classes,
        layers,
        criterion,
        steps=4,
        multiplier=4,
        stem_multiplier=3,
        device="cuda",
    ):
        super(MacroNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self._multiplier = multiplier

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False), nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C

        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(
                steps,
                multiplier,
                C_prev_prev,
                C_prev,
                C_curr,
                reduction,
                reduction_prev,
            )
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas(device)

    def new(self):
        model_new = Network(
            self._C, self._num_classes, self._layers, self._criterion
        ).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.reduction:
                weights = F.softmax(self.alphas_reduce, dim=-1)
            else:
                weights = F.softmax(self.alphas_normal, dim=-1)
            s0, s1 = s1, cell(s0, s1, weights)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target)

    def _initialize_alphas(self, device):
        k = sum(1 for i in range(self._steps) for n in range(2 + i))
        num_ops = len(PRIMITIVES)

        if device == "cuda":
            self.alphas_normal = Variable(
                1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
            )
            self.alphas_reduce = Variable(
                1e-3 * torch.randn(k, num_ops).cuda(), requires_grad=True
            )
        else:
            self.alphas_normal = Variable(
                1e-3 * torch.randn(k, num_ops), requires_grad=True
            )
            self.alphas_reduce = Variable(
                1e-3 * torch.randn(k, num_ops), requires_grad=True
            )
        self._arch_parameters = [
            self.alphas_normal,
            self.alphas_reduce,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def build_architecture(
        self, dataset_name, arch_dict, init_channels, layers, auxiliary, selected_layers
    ):
        if dataset_name == "cifar10":
            CIFAR_CLASSES = 10
            arch_dict.update(
                {
                    "model": HeterogenousNetworkCIFAR(
                        init_channels, CIFAR_CLASSES, layers, auxiliary, selected_layers
                    )
                }
            )
        elif dataset_name == "imagenet":
            IMAGENET_CLASSES = 1000
            arch_dict.update(
                {
                    "model": HeterogenousNetworkImageNet(
                        init_channels,
                        IMAGENET_CLASSES,
                        layers,
                        auxiliary,
                        selected_layers,
                    )
                }
            )

        return arch_dict

    def sample_architecture(
        self,
        dataset_name,
        nsample,
        micro_genotypes,
        init_channels,
        max_layers,
        n_family,
        auxiliary,
    ):
        architectures = []
        valid_gen_choice = micro_genotypes["genotype"].tolist()
        valid_gen_choice.append("none")
        ln_valid_choice = len(valid_gen_choice) - 1
        for ifamily in range(1, n_family + 1):
            valid_layer = int(ifamily * max_layers / n_family)
            print("val layer: {}".format(valid_layer))
            for _ in range(nsample):
                selected_layers = []
                selected_idx = []
                for i in range(valid_layer):
                    rand_idx = random.randint(0, ln_valid_choice)
                    selected_idx.append(rand_idx)
                    selected_layers.append(valid_gen_choice[rand_idx])
                none_layers = [i for i, x in enumerate(selected_layers) if x == "none"]
                name = ";".join([str(elem) for elem in selected_layers])
                skip_conn = [i.start() for i in re.finditer("skip", name)]
                arch_dict = {
                    "selected_medioid_idx": selected_idx,
                    "cell_layers": valid_layer - len(none_layers),
                    "none_layers": len(none_layers),
                    "skip_conn": len(skip_conn),
                    "name": name,
                }

                architectures.append(
                    self.build_architecture(
                        dataset_name,
                        arch_dict,
                        init_channels,
                        valid_layer,
                        auxiliary,
                        selected_layers,
                    )
                )

        return architectures

    def find_target_latency(
        self,
        dataset_name,
        micro_genotype,
        init_channels,
        max_layers,
        auxiliary,
        drop_path_prob,
        n_family,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        max_lat_genotypes = [micro_genotype for i in range(max_layers)]
        if dataset_name == "cifar10":
            INPUT_BATCH = 1
            INPUT_CHANNEL = 3
            INPUT_SIZE = 32
            CLASSES = 10
            model = HeterogenousNetworkCIFAR(
                init_channels, CLASSES, max_layers, auxiliary, max_lat_genotypes
            )

        elif dataset_name == "imagenet":
            INPUT_BATCH = 1
            INPUT_CHANNEL = 3
            INPUT_SIZE = 224
            CLASSES = 1000
            layer_depth = 25
            model = HeterogenousNetworkImageNet(
                init_channels, CLASSES, max_layers, auxiliary, max_lat_genotypes
            )

        else:
            INPUT_BATCH = 1
            INPUT_CHANNEL = 3
            INPUT_SIZE = 32
            CLASSES = 10
            layer_depth = 40
            model = HeterogenousNetworkCIFAR(
                init_channels, CLASSES, layer_depth, auxiliary, max_lat_genotypes
            )

        model.drop_path_prob = drop_path_prob
        model.to(device)
        dummy_input = torch.zeros(
            INPUT_BATCH, INPUT_CHANNEL, INPUT_SIZE, INPUT_SIZE
        ).to(device)
        mean_lat, latencies = latency_profiler.test_latency(model, dummy_input, device)
        print("biggest lat ", latencies[98])
        partition_lat = latencies[98] / n_family
        target_latency = [partition_lat * (i + 1) for i in range(n_family)]
        print("target_latency ", target_latency)
        return target_latency

    def convert_node_to_arch_dict(self, current_node, selected_layers):
        print("state: ln {} moves {}".format(len(selected_layers), selected_layers))
        name = ";".join([str(elem) for elem in selected_layers])
        none_layers = [i for i, x in enumerate(selected_layers) if x == "none"]
        cell_layers = len(selected_layers) - len(none_layers)
        skip_conn = [i.start() for i in re.finditer("skip", name)]
        return {
            "selected_medioid_idx": current_node.state.selected_med_idx,
            "cell_layers": cell_layers,
            "none_layers": len(none_layers),
            "skip_conn": len(skip_conn),
            "name": name,
        }

    def sample_mcts_architecture(
        self,
        dataset_name,
        nsample,
        micro_genotypes,
        init_channels,
        max_layers,
        n_family,
        auxiliary,
        drop_path_prob,
    ):
        architectures = []
        config = {
            "architecture": {
                "init_channels": init_channels,
                "num_classes": 10,
                "layers": max_layers,
                "auxiliary": auxiliary,
                "drop_path_prob": drop_path_prob,
            },
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "lr": 0.025,
            "momentum": 0.9,
        }

        num_sims = 100
        n_turn = n_family + 2
        target_latency = self.find_target_latency(
            dataset_name,
            micro_genotypes["genotype"].to_list()[-1],
            init_channels,
            max_layers,
            auxiliary,
            drop_path_prob,
            n_family,
        )
        current_node = mcts.Node(
            mcts.State(
                moves=[],
                selected_med_idx=[],
                turn=n_turn,
                n_family=n_family,
                target_latency=target_latency,
                max_layers=max_layers,
                config=config,
            )
        )

        for fam_member in range(n_family):
            current_nodes = mcts.UCTSEARCH(num_sims / (fam_member + 1), current_node)
            if isinstance(current_nodes, list):
                # Use the list of top-k children from the tree to create architecture
                print(
                    "fam_member ",
                    fam_member,
                    " len(current_nodes) ",
                    len(current_nodes),
                )
                for node in current_nodes:
                    selected_layers = node.state.moves
                    arch_dict = self.convert_node_to_arch_dict(node, selected_layers)
                    architectures.append(
                        self.build_architecture(
                            dataset_name,
                            arch_dict,
                            init_channels,
                            arch_dict["cell_layers"],
                            auxiliary,
                            selected_layers,
                        )
                    )
                # Use the best reward simulation to continue searching as root
                current_node = current_nodes[0]

        print(len(architectures), ";;; arch ")
        return architectures
