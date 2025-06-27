""" Config class for search/augment - Updated with MaxK kernel options """
import argparse
import os
from functools import partial


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser

class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)

        return text

class TrainConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("GIN/GCN/GraphSAGE Training config with MaxK acceleration")
        parser.add_argument('--dataset',  default='yelp', choices = ['reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins'], #, 
                            help="Dataset name ('reddit', 'flickr', 'yelp', 'ogbn-arxiv', 'ogbn-products', 'ogbn-proteins').") #, 'ogbn-proteins'
        parser.add_argument('--data_path', default='./data/', help='Dataset path')
        parser.add_argument('--model', default='sage', type=str, choices = ['sage', 'gcn', 'gin', 'gnn_res'],
                                    help="Model used in the training ('sage', 'gcn', 'gin', 'gnn_res')")
        parser.add_argument('--selfloop', default=False, action='store_true', help='add selfloop or not') #5e-4
        parser.add_argument('--epochs', type=int, default=1000, help='# of training epochs')
        parser.add_argument('--w_lr', type=float, default=0.01, help='lr for weights')
        parser.add_argument('--w_weight_decay', type=float, default=0, help='weight decay for weights') #5e-4
        parser.add_argument('--enable_lookahead', default=False, action='store_true', help='Using lookahead optimizer or not')
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help='Hidden dimension size')
        parser.add_argument('--hidden_layers', default=3, type=int,
                            help='Hidden dimension layers')
        parser.add_argument('--nonlinear', default='maxk', type=str, choices = ['maxk', 'relu'],
                            help='Nonlinear function used in the model')
        parser.add_argument('--maxk', default=32, type=int,
                            help='k value for maxk non-linearity')
        parser.add_argument('--dropout', type=float, default=0.5, help='feature dropout ratio') #5e-4
        parser.add_argument('--norm', default=False, action='store_true', help='add normalization layer or not') #5e-4
        parser.add_argument('--gpu', type=int, default=0, help='gpu device used in the experiment')
        parser.add_argument('--seed', type=int, default=97, help='random seed')
        parser.add_argument('-e', '--evaluate', default=None, type=str, metavar='PATH',
                            help='path to checkpoint (default: none), evaluate model')
        parser.add_argument('--path', default='./run/', type=str, metavar='PATH',
                            help='path to save the model and logging')
        
        # NEW: MaxK kernel acceleration options
        parser.add_argument('--use_maxk_kernels', default=False, action='store_true', 
                            help='Use MaxK CUDA kernels for SpGEMM acceleration')
        parser.add_argument('--kernel_mode', default='auto', type=str, 
                            choices=['auto', 'maxk', 'cusparse', 'dgl'],
                            help='Kernel mode: auto (try MaxK, fallback), maxk (MaxK only), cusparse (cuSPARSE), dgl (DGL default)')
        parser.add_argument('--graph_metadata_path', default='kernels/w12_nz64_warp_4/', 
                            type=str, help='Path to warp4 metadata files')
        parser.add_argument('--validate_kernels', default=False, action='store_true',
                            help='Validate MaxK kernel correctness against cuSPARSE')
        parser.add_argument('--profile_kernels', default=False, action='store_true',
                            help='Profile kernel performance during training')
        parser.add_argument('--save_model', default=False, action='store_true',
                            help='Save the final trained model')
        
        # Advanced MaxK options
        parser.add_argument('--maxk_num_warps', default=12, type=int,
                            help='Number of warps for MaxK kernels')
        parser.add_argument('--maxk_warp_max_nz', default=64, type=int,
                            help='Maximum non-zeros per warp for MaxK kernels')
        parser.add_argument('--maxk_fallback_threshold', default=0.001, type=float,
                            help='Error threshold for falling back from MaxK to cuSPARSE')

        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.plot_path = os.path.join(self.path, 'plots')
        
        # Validate MaxK options
        if self.use_maxk_kernels and self.kernel_mode == 'dgl':
            print("⚠️ Warning: use_maxk_kernels=True but kernel_mode=dgl, using DGL fallback")
            self.use_maxk_kernels = False
        
        if self.maxk > 256:
            print(f"⚠️ Warning: maxk={self.maxk} is very large, may cause memory issues")
        
        if self.use_maxk_kernels and not os.path.exists(self.graph_metadata_path):
            print(f"⚠️ Warning: metadata path {self.graph_metadata_path} does not exist")
            print("   Run 'python kernels/generate_meta.py' to generate metadata")
