CUDA_VISIBLE_DEVICES=0 python3 train_transductive.py --flagfile=config/wiki-cs.cfg --drop_edge_p=0.9 --drop_feat_p=0.2 --lr=5e-4 --weight_decay=1e-5 --lr_cls=2e-2
CUDA_VISIBLE_DEVICES=0 python3 train_transductive.py --flagfile=config/amazon-computers.cfg --drop_edge_p=0.9 --drop_feat_p=0.2 --lr=5e-4 --weight_decay=5e-4 --lr_cls=5e-2
CUDA_VISIBLE_DEVICES=0 python3 train_transductive.py --flagfile=config/amazon-photos.cfg --drop_edge_p=0.9 --drop_feat_p=0.2 --lr=1e-4 --weight_decay=1e-4 --lr_cls=5e-2
CUDA_VISIBLE_DEVICES=0 python3 train_transductive.py --flagfile=config/coauthor-cs.cfg --drop_edge_p=0.5 --drop_feat_p=0.4 --lr=1e-5 --weight_decay=1e-5 --lr_cls=4e-2
CUDA_VISIBLE_DEVICES=0 python3 train_transductive.py --flagfile=config/coauthor-physics.cfg --drop_edge_p=0.8 --drop_feat_p=0.6 --lr=1e-4 --weight_decay=1e-5 --lr_cls=1e-1
CUDA_VISIBLE_DEVICES=0 python3 train_transductive.py --flagfile=config/arxiv.cfg --drop_edge_p=0.8 --drop_feat_p=0 --lr=1e-4 --weight_decay=1e-5 --wd_cls=5e-6
CUDA_VISIBLE_DEVICES=1 python3 train_transductive_mag.py --flagfile=config/mag.cfg --drop_edge_p=0.8 --drop_feat_p=0 --weight_decay=1e-4 --wd_cls=0 --lr=1e-4
CUDA_VISIBLE_DEVICES=1 python3 train_transductive_products.py --flagfile=config/products.cfg --drop_edge_p=0.4 --drop_feat_p=0 --weight_decay=1e-5 --wd_cls=5e-6 --lr=1e-2
