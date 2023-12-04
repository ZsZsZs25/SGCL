import copy
import logging
import os
from statistics import mode

from absl import app
from absl import flags
import numpy as np
import torch
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from tqdm import tqdm

from bgrl import *
from bgrl.utils import edgeidx2sparse

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', 123, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
# Dataset.
flags.DEFINE_enum('dataset', 'WikiCS',
                  ['Cora', 'Citeseer', 'Pubmed', 'Computers', 'Photos', 'CS', 'Physics', 'WikiCS', 'arxiv'],
                  'Which graph dataset to use.')
flags.DEFINE_string('dataset_dir', '~/data/pygdata/', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', [512, 256], 'Conv layer sizes.')
flags.DEFINE_bool('batchnorm', True, 'Batchnorm or not.')
flags.DEFINE_bool('layernorm', False, 'Layernorm or not.')
flags.DEFINE_bool('weight_standardization', False, 'Weight Standardization or not.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 10, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_integer('lr_warmup_epochs', 1000, 'Warmup period for learning rate.')

flags.DEFINE_float('lr_cls', 1e-5, 'The learning rate for model training for node classification classifier..')
flags.DEFINE_float('wd_cls', 1e-5, 'The value of the weight decay for training for node classification classifier..')
flags.DEFINE_integer('epochs_cls', 100, 'The number of training epochs for node classification classifier.')

# Augmentations.
flags.DEFINE_float('drop_edge_p', 0., 'Probability of edge dropout 1.')
flags.DEFINE_float('drop_feat_p', 0., 'Probability of node feature dropout 1.')
flags.DEFINE_float('epsilon', 0., 'Probability of node feature dropout 1.')

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_string('maskdir', './mask', 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 5, 'Evaluate every eval_epochs.')


def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # device = torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # create log directory
    if FLAGS.logdir is not None:
        log.info("Logdir: {}".format(FLAGS.logdir))
        os.makedirs(FLAGS.logdir, exist_ok=True)
        with open(os.path.join(FLAGS.logdir, '{}.cfg'.format(FLAGS.dataset)), "w") as file:
            file.write(FLAGS.flags_into_string())  # save config file

    # load data
    if FLAGS.dataset != 'WikiCS':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_classes = dataset.num_classes
    else:
        dataset, num_classes = get_wiki_cs(FLAGS.dataset_dir +"/Wiki-CS")        

    # load mask
    if FLAGS.maskdir is not None:
        log.info("Preset mask dir: {}".format(FLAGS.maskdir))
        os.makedirs(FLAGS.maskdir, exist_ok=True)
        mask_path = "{}/{}_mask.pt".format(FLAGS.maskdir, FLAGS.dataset)
        if not os.path.exists(mask_path):
            train_mask, val_mask, test_mask = create_mask(dataset, FLAGS.dataset, FLAGS.data_seed, mask_path)
            log.info("Preset mask for dataset {} not exists. Creatting Now.".format(FLAGS.dataset))
        else:
            train_mask, val_mask, test_mask = torch.load(mask_path)
            log.info("Preset mask load from {}.".format(mask_path))
    else:
        log.info("Preset mask dir not specified. Create Now.")
        train_mask, val_mask, test_mask = create_mask(dataset, FLAGS.dataset, FLAGS.data_seed, mask_path='tmp.pt')

    data = dataset[0]  # all dataset include one graph
    # set random seed
    if FLAGS.model_seed is not None:
        log.info('Random seed set to {}.'.format(FLAGS.model_seed))
        set_random_seeds(random_seed=FLAGS.model_seed)
    
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    if FLAGS.dataset in ['arxiv']:
        data.y = data.y.squeeze()
    data = data.to(device)  # permanently move in gpy memory
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))

    # prepare transforms
    transform = get_graph_drop_transform(drop_edge_p=FLAGS.drop_edge_p, drop_feat_p=FLAGS.drop_feat_p)

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=FLAGS.batchnorm, 
                layernorm=FLAGS.layernorm, weight_standardization=FLAGS.weight_standardization)   # 512, 256, 128
    predictor = Predictor()
    model = BGRL(encoder, predictor).to(device)
    # log.info(model)
    # optimizer
    optimizer = AdamW(model.trainable_parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)

    # number of parameters    
    total = sum([param.nelement() for param in model.parameters()])
    
    log.info("Number of parameter: %.2fM" % (total/1e6))
    

    def train(step, target):
        model.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # forward
        optimizer.zero_grad()

        x = transform(data)
        x.edge_index = edgeidx2sparse(x.edge_index, x.x.size(0))
        
        _, online = model(x, target)
        
        loss = 1 - cosine_similarity(online, target.detach(), dim=-1).mean()

        loss.backward()
        
        # update online network
        optimizer.step()

        # # next target
        model.eval()
        with torch.no_grad():
            target = model.online_representation(x)

        return lr, loss.item(), target.detach()

    def eval(epoch):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(model.online_encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)
        
        if FLAGS.dataset in ['arxiv', 'Cora', 'Citeseer', 'Pubmed']:
            test_acc_list = node_cls_downstream_task_eval(representations, data, num_classes, 
                                                FLAGS.lr_cls, FLAGS.wd_cls, cls_runs=10, 
                                                cls_epochs=FLAGS.epochs_cls, device=device)
        else:
            test_acc_list = node_cls_downstream_task_multi_eval(representations, data, num_classes, 
                                                FLAGS.lr_cls, cls_epochs=FLAGS.epochs_cls, device=device)

        return np.mean(test_acc_list), np.std(test_acc_list), test_acc_list

    # augmentation first to obtain a random target
    x = transform(data)
    x.edge_index = edgeidx2sparse(x.edge_index, x.x.size(0))
    with torch.no_grad():
        target = model.online_representation(x).detach()

    best_test_acc_mean, best_test_acc_std, best_test_acc_epoch, best_test_acc_list = 0, 0, 0, []

    for epoch in range(1, FLAGS.epochs + 1):
        lr, loss, target = train(epoch, target)

        if epoch == 1 or epoch % FLAGS.eval_epochs == 0:
            
            # test_acc_mean, test_acc_std, test_acc_list = 0, 0, []
            test_acc_mean, test_acc_std, test_acc_list = eval(epoch)
            if test_acc_mean > best_test_acc_mean:
                best_test_acc_mean = test_acc_mean
                best_test_acc_std = test_acc_std
                best_test_acc_epoch = epoch
                best_test_acc_list = copy.deepcopy(test_acc_list)
                # save encoder weights
                # torch.save(model.online_encoder.state_dict(), os.path.join(FLAGS.logdir, '{}.pt'.format(FLAGS.dataset)))

            log.info("[Epoch {:4d}/{:4d}] lr={:.4f}, loss={:.4f}, test_acc={:.2f}±{:.2f} [best_test_acc: {:.2f}±{:.2f} at epoch {}]".format(
                epoch, FLAGS.epochs + 1, lr, loss, test_acc_mean * 100, test_acc_std * 100, 
                best_test_acc_mean * 100, best_test_acc_std * 100, best_test_acc_epoch
            ))

    log.info("Best test acc: {:.2f}±{:.2f} at epoch {}: {}".format(
        best_test_acc_mean * 100, best_test_acc_std * 100, best_test_acc_epoch, best_test_acc_list
    ))


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
