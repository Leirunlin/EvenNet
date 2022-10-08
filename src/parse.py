from models import *


def parser_add_main_args(parser):
    # hyperparameters for training
    parser.add_argument('--seed', type=int, default=15, help='seeds for random splits.')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs.')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay.')
    parser.add_argument('--early_stopping', type=int, default=200, help='early stopping.')
    parser.add_argument('--hidden', type=int, default=64, help='hidden units.')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout for neural networks.')
    parser.add_argument('--train_rate', type=float, default=0.1, help='train percent')
    parser.add_argument('--val_rate', type=float, default=0.1, help='val percent')

    # hyperparameters for networks
    parser.add_argument('--K', type=int, default=10, help='propagation steps.')
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha for EvenNet.')
    parser.add_argument('--dprate', type=float, default=0.5, help='dropout for propagation layer.')
    parser.add_argument('--Init', type=str, default='PPR', help='initialization for EvenNet.')
    parser.add_argument('--heads', default=8, type=int, help='attention heads for GAT.')
    parser.add_argument('--pro_lr', type=float, default=0.01, help='learning rate for propagation layer.')
    parser.add_argument('--ppnp', type=str, default='GPR_prop', help='GPRGNN')
    parser.add_argument('--num_layers', type=int, default=32, help='num of layers')
    parser.add_argument('--num_stacks', type=int, default=2, help='num of stacks')
    parser.add_argument('--gcn2_alpha', type=float, default=.2, help='alpha for gcn2')
    parser.add_argument('--theta', type=float, default=.5, help='theta for gcn2')
    parser.add_argument('--eps', type=float, default=.3, help='eps for FAGCN')
    parser.add_argument('--gamma', type=float, default=1.0, help='gamma for RGCN')
    parser.add_argument('--beta1', type=float, default=5e-4, help='beta1 for RGCN')
    parser.add_argument('--beta2', type=float, default=5e-4, help='beta2 for RGCN')
    parser.add_argument('--k_SVD', type=int, default=5, help='Rank for GCN_SVD')

    parser.add_argument('--mu', type=float, default=0.1, help='mu for pGNN')
    parser.add_argument('--p', type=float, default=1.5, help='p for pGNN')
    parser.add_argument('--threshold', type=float, default=0.1, help='t for Guard')

    # hyperparameters for attacks
    parser.add_argument('--attack_type', type=str, choices=['DICE', 'Meta', 'MinMax', 'Random'], default="DICE")
    parser.add_argument('--attack_ratio', type=float, default=0.2)

    # Other parameters
    parser.add_argument('--device', type=int, default=0, help='GPU device.')
    parser.add_argument('--runs', type=int, default=5, help='number of runs.')
    parser.add_argument('--net', type=str, choices=['GCN', 'GCN_Jaccard', 'GCN_SVD', "RGCN", "GCN_Guard",
                                                    "EGCN_Guard", 'GAT', "FAGCN", 'GCNII', 'H2GCN',
                                                    'GPRGNN', 'GPRGNN_reg', 'EvenNet', "pGNN",
                                                    'BernNet', 'MLP'], default='GCN')
    parser.add_argument('--dataset', type=str, default='cora')


def parse_method(args, dataset):
    net_name = args.net
    device = args.device
    in_channels = dataset.num_features
    out_channels = dataset.num_classes
    hidden_size = args.hidden
    dropout = args.dropout

    if net_name == "MLP":
        model = MLP(in_channels=in_channels, out_channels=out_channels,
                    hidden_size=hidden_size, dropout=dropout).to(device)
    elif net_name == "GCN":
        model = GCN(in_channels=in_channels, out_channels=out_channels,
                    hidden_size=hidden_size, dropout=dropout).to(device)
    elif net_name == "GAT":
        heads = args.heads
        model = GAT(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                    heads=heads, dropout=dropout).to(device)
    elif net_name == "GCNII":
        num_layers = args.num_layers
        theta = args.theta
        alpha = args.gcn2_alpha
        model = GCNII(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                      num_layers=num_layers, alpha=alpha, lamda=theta,
                      dropout=dropout).to(device)
    elif net_name == "FAGCN":
        num_layers = args.num_layers
        eps = args.eps
        model = FAGCN(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                      layer_num=num_layers, eps=eps,
                      dropout=dropout).to(device)
    elif net_name == "H2GCN":
        n = dataset[0].num_nodes
        model = H2GCN(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                    num_nodes=n, num_layers=args.num_layers, dropout=dropout).to(device)
    elif "GPRGNN" in net_name:
        k = args.K
        init = args.Init
        alpha = args.alpha
        dprate = args.dprate
        if net_name == "GPRGNN" :
            model = GPRGNN(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                           K=k, Init=init, alpha=alpha, dprate=dprate, dropout=dropout).to(device)
        elif net_name == "GPRGNN_reg":
            model = GPRGNN_reg(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                           K=k, Init=init, alpha=alpha, dprate=dprate, dropout=dropout).to(device)
    elif net_name == 'EvenNet':
        k = args.K
        init = args.Init
        alpha = args.alpha
        dprate = args.dprate
        model = EvenNet(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                        K=k, Init=init, alpha=alpha, dprate=dprate, dropout=dropout).to(device)
    elif "BernNet" in net_name:
        k = args.K
        dprate = args.dprate
        model = BernNet(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                        k=k, dprate=dprate, dropout=dropout).to(device)

    elif net_name == 'pGNN':
        model = pGNNNet(in_channels=in_channels,
                    out_channels=out_channels,
                    num_hid=hidden_size,
                    mu=args.mu,
                    p=args.p,
                    K=args.K,
                    dropout=dropout)
    elif net_name == "RGCN":
        n_nodes = dataset[0].num_nodes
        model = RGCN(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                     device=device, n_nodes=n_nodes, dropout=dropout, gamma=args.gamma,
                     beta1=args.beta1, beta2=args.beta2).to(device)
    elif net_name == "GCN_Jaccard":
        model = GCN_Jaccard(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                            device=device, dropout=dropout).to(device)
    elif net_name == "GCN_SVD":
        model = GCN_SVD(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                        device=device, dropout=dropout, k=args.k_SVD).to(device)
    elif net_name == "GCN_Guard":
        model = GCN_Guard(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                          device=device, dropout=dropout, attention=True).to(device)
    elif net_name == "EGCN_Guard":
        model = EGCNGuard(in_channels=in_channels, out_channels=out_channels, hidden_size=hidden_size,
                          dropout=dropout, threshold=args.threshold).to(device)

    else:
        raise ValueError('Invalid method')
    return model
