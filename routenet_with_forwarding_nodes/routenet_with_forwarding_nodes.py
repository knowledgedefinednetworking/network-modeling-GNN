# Copyright (c) 2019, Krzysztof Rusek [^1], Paul Almasan [^2], Arnau Badia [^3]
#
# [^1]: AGH University of Science and Technology, Department of
#     communications, Krakow, Poland. Email: krusek\@agh.edu.pl
#
# [^2]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: almasan@ac.upc.edu
#
# [^3]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: abadia@ac.upc.edu


from __future__ import print_function

import numpy as np
import pandas as pd
import networkx as nx
import itertools as it
import os
import tensorflow as tf
from tensorflow import keras
import re
import argparse
import random
import tarfile
import glob

def genPath(R,s,d,connections):
    while s != d:
        yield s
        s = connections[s][R[s,d]]
    yield s

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = it.tee(iterable)
    next(b, None)
    return zip(a, b)

def load_routing(routing_file):
    R = pd.read_csv(routing_file, header=None, index_col=False)
    R=R.drop([R.shape[0]], axis=1)
    return R.values

def make_indices(paths):
    # node can be understood as either "node entity" or "link entity", it is used for both
    node_indices=[]
    path_indices=[]
    sequ_indices=[]
    segment=0
    for p in paths:
        node_indices += p
        path_indices += len(p)*[segment]
        sequ_indices += list(range(len(p)))
        segment +=1
    return node_indices, path_indices, sequ_indices

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_features(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _float_features(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def parse(serialized, target='delay'):
    '''
    Target is the name of predicted variable
    '''
    with tf.device("/cpu:0"):
        with tf.name_scope('parse'):
            features = tf.parse_single_example(
                serialized,
                features={
                    'traffic':tf.VarLenFeature(tf.float32),
                    target:tf.VarLenFeature(tf.float32),
                    'link_capacity': tf.VarLenFeature(tf.float32),
                    'queue_sizes': tf.VarLenFeature(tf.float32),
                    'links':tf.VarLenFeature(tf.int64),
                    'link_paths':tf.VarLenFeature(tf.int64),
                    'link_sequences':tf.VarLenFeature(tf.int64),
                    'nodes':tf.VarLenFeature(tf.int64),
                    'node_paths':tf.VarLenFeature(tf.int64),
                    'node_sequences':tf.VarLenFeature(tf.int64),
                    'n_links':tf.FixedLenFeature([],tf.int64),
                    'n_nodes':tf.FixedLenFeature([],tf.int64),
                    'n_paths':tf.FixedLenFeature([],tf.int64),
                    'n_link_total':tf.FixedLenFeature([],tf.int64),
                    'n_node_total':tf.FixedLenFeature([],tf.int64)
                })
            for k in ['traffic',target,'link_capacity','queue_sizes','links','link_paths',
                      'link_sequences','nodes','node_paths','node_sequences']:
                features[k] = tf.sparse_tensor_to_dense(features[k])
                if k == 'delay':
                    features[k] = (tf.math.log(features[k]) + 1.78) / 0.93
                if k == 'traffic':
                    features[k] = (features[k] - 0.28) / 0.15
                if k == 'jitter':
                    features[k] = (features[k] - 1.5) / 1.5
                if k == 'link_capacity':
                    features[k] = (features[k] - 27.0) / 14.86
                if k == 'queue_sizes':
                    features[k] = (features[k] - 16.5) / 15.5

    return {k:v for k,v in features.items() if k is not target },features[target]

def cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max( extractor(v) ) + 1 for v in alist ]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes)-1):
            cummaxes.append( tf.math.add_n(maxes[0:i+1]))

    return cummaxes


def transformation_func(it, batch_size=32):
    with tf.name_scope("transformation_func"):
        vs = [it.get_next() for _ in range(batch_size)]

        links_cummax = cummax(vs,lambda v:v[0]['links'])
        link_paths_cummax = cummax(vs,lambda v:v[0]['link_paths'])
        nodes_cummax = cummax(vs,lambda v:v[0]['nodes'])
        node_paths_cummax = cummax(vs,lambda v:v[0]['node_paths'])

        tensors = ({
                'traffic':tf.concat([v[0]['traffic'] for v in vs], axis=0),
                'link_capacity': tf.concat([v[0]['link_capacity'] for v in vs], axis=0),
                'queue_sizes': tf.concat([v[0]['queue_sizes'] for v in vs], axis=0),
                'links':tf.concat([v[0]['links'] + m for v,m in zip(vs, links_cummax) ], axis=0),
                'link_paths':tf.concat([v[0]['link_paths'] + m for v,m in zip(vs, link_paths_cummax) ], axis=0),
                'link_sequences':tf.concat([v[0]['link_sequences'] for v in vs], axis=0),
                'nodes':tf.concat([v[0]['nodes'] + m for v,m in zip(vs, nodes_cummax) ], axis=0),
                'node_paths':tf.concat([v[0]['node_paths'] + m for v,m in zip(vs, node_paths_cummax) ], axis=0),
                'node_sequences':tf.concat([v[0]['node_sequences'] for v in vs], axis=0),
                'n_links':tf.math.add_n([v[0]['n_links'] for v in vs]),
                'n_nodes':tf.math.add_n([v[0]['n_nodes'] for v in vs]),
                'n_paths':tf.math.add_n([v[0]['n_paths'] for v in vs]),
                'n_link_total':tf.math.add_n([v[0]['n_link_total'] for v in vs]),
                'n_node_total':tf.math.add_n([v[0]['n_node_total'] for v in vs])
            },   tf.concat([v[1] for v in vs], axis=0))

    return tensors


def tfrecord_input_fn(filenames,hparams,shuffle_buf=1000,target='delay'):

    files = tf.data.Dataset.from_tensor_slices(filenames)
    files = files.shuffle(len(filenames))

    ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    if shuffle_buf:
        ds = ds.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buf))

    ds = ds.map(lambda buf:parse(buf,target),
        num_parallel_calls=2)
    ds=ds.prefetch(10)

    it =ds.make_one_shot_iterator()
    sample = transformation_func(it,hparams.batch_size)


    return sample

class ComnetModel(tf.keras.Model):
    def __init__(self,hparams, output_units=1, final_activation=None):
        super(ComnetModel, self).__init__()
        self.hparams = hparams

        self.edge_update = tf.keras.layers.GRUCell(hparams.link_state_dim)
        self.node_update = tf.keras.layers.GRUCell(hparams.link_state_dim)
        self.path_update = tf.keras.layers.GRUCell(hparams.path_state_dim)


        self.readout = tf.keras.models.Sequential()

        self.readout.add(keras.layers.Dense(hparams.readout_units,
                activation=tf.nn.selu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))
        self.readout.add(keras.layers.Dense(hparams.readout_units,
                activation=tf.nn.selu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2)))
        self.readout.add(keras.layers.Dropout(rate=hparams.dropout_rate))

        self.readout.add(keras.layers.Dense(output_units,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(hparams.l2_2),
                activation = final_activation ) )


    def build(self, input_shape=None):
        del input_shape
        self.edge_update.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.node_update.build(tf.TensorShape([None,self.hparams.path_state_dim]))
        self.path_update.build(tf.TensorShape([None,self.hparams.link_state_dim]))
        self.readout.build(input_shape = [None,self.hparams.path_state_dim])
        self.built = True


    def call(self, inputs, training=False):
        f_ = inputs

        shape = tf.stack([f_['n_links'], self.hparams.link_state_dim-1], axis=0)
        link_state = tf.concat([
            tf.expand_dims(f_['link_capacity'], axis=1),
            tf.zeros(shape)
        ], axis=1)
        shape = tf.stack([f_['n_nodes'], self.hparams.link_state_dim-1], axis=0)
        node_state = tf.concat([
            tf.expand_dims(f_['queue_sizes'], axis=1),
            tf.zeros(shape)
        ], axis=1)

        shape = tf.stack([f_['n_paths'],self.hparams.path_state_dim-1], axis=0)
        path_state = tf.concat([
            tf.expand_dims(f_['traffic'][0:f_["n_paths"]],axis=1),
            tf.zeros(shape)
        ], axis=1)

        links = f_['links']
        link_paths = f_['link_paths']
        link_seqs =  f_['link_sequences']
        nodes = f_['nodes']
        node_paths = f_['node_paths']
        node_seqs = f_['node_sequences']

        for _ in range(self.hparams.T):

            """
            PART 1: Inference from node and link states to path states
            """

            # build link_inputs
            link_h_tild = tf.gather(link_state, links)
            link_ids=tf.stack([link_paths, link_seqs], axis=1)
            link_paths_max_len = tf.reduce_max(link_seqs)+1
            link_inputs_shape = tf.stack([f_['n_paths'], link_paths_max_len,
                                          self.hparams.link_state_dim])
            link_lens = tf.segment_sum(data=tf.ones_like(link_paths),
                                    segment_ids=link_paths)
            link_inputs = tf.scatter_nd(link_ids, link_h_tild, link_inputs_shape)

            # build node inputs
            node_h_tild = tf.gather(node_state, nodes)
            node_ids=tf.stack([node_paths, node_seqs], axis=1)
            node_paths_max_len = tf.reduce_max(node_seqs)+1
            node_inputs_shape = tf.stack([f_['n_paths'], node_paths_max_len,
                                          self.hparams.link_state_dim])
            node_lens = tf.segment_sum(data=tf.ones_like(node_paths),
                                    segment_ids=node_paths)
            node_inputs = tf.scatter_nd(node_ids, node_h_tild, node_inputs_shape)

            # create node_link_inputs by interleaving the corresponding
            # sequence of node-link-node-link-...-node
            c = tf.concat([node_inputs, link_inputs], axis=1)
            c = tf.transpose(c, perm=[1,0,2])
            total_paths_max_len = node_paths_max_len + link_paths_max_len
            c_indices = tf.concat([tf.range(0,total_paths_max_len,2),
                                   tf.range(1,total_paths_max_len,2)], axis=0)
            c_indices = tf.reshape(c_indices, [-1,1])
            c = tf.scatter_nd(c_indices, c, tf.shape(c, out_type=tf.int64))
            c = tf.transpose(c, perm=[1,0,2])
            node_link_inputs = c

            # calculate combined path lengths
            lens = tf.math.add(link_lens, node_lens)

            outputs, path_state = tf.nn.dynamic_rnn(self.path_update,
                                                    node_link_inputs,
                                                    sequence_length=lens,
                                                    initial_state = path_state,
                                                    dtype=tf.float32)

            """
            PART 2: Inference from path states to node and link states
            """

            # separate between node_outputs (even indices) and link_outputs (odd indices)
            node_outputs = outputs[:,0::2,:]
            link_outputs = outputs[:,1::2,:]

            # update links from path information
            m_links = tf.gather_nd(link_outputs,link_ids)
            m_links = tf.math.unsorted_segment_sum(m_links, links ,f_['n_links'])
            #Keras cell expects a list
            link_state,_ = self.edge_update(m_links, [link_state])

            # update nodes from path information
            m_nodes = tf.gather_nd(node_outputs,node_ids)
            m_nodes = tf.math.unsorted_segment_sum(m_nodes, nodes ,f_['n_nodes'])
            node_state, _ = self.node_update(m_nodes, [node_state])


        if self.hparams.learn_embedding:
            r = self.readout(path_state,training=training)
        else:
            r = self.readout(tf.stop_gradient(path_state),training=training)

        return r

def model_fn(
   features, # This is batch_features from input_fn
   labels,   # This is batch_labrange
   mode,     # An instance of tf.estimator.ModeKeys
   params):  # Additional configuration


    model = ComnetModel(params)
    model.build()

    def fn(x):
        r = model(x,training=mode==tf.estimator.ModeKeys.TRAIN)
        return r

    predictions = fn(features)

    predictions = tf.squeeze(predictions)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,
            predictions={'predictions':predictions})

    loss =  tf.losses.mean_squared_error(
        labels=labels,
        predictions = predictions,
        reduction=tf.losses.Reduction.MEAN
    )

    regularization_loss = sum(model.losses)
    total_loss = loss + regularization_loss

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('regularization_loss', regularization_loss)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss,
            eval_metric_ops={
                'label/mean': tf.metrics.mean(labels),
                'prediction/mean': tf.metrics.mean(predictions),
                'mae': tf.metrics.mean_absolute_error(labels, predictions),
                'rho': tf.contrib.metrics.streaming_pearson_correlation(labels=labels, predictions=predictions),
                'mre': tf.metrics.mean_relative_error(labels, predictions, labels)
            }
        )

    assert mode == tf.estimator.ModeKeys.TRAIN


    trainables = model.variables
    grads = tf.gradients(total_loss, trainables)
    grad_var_pairs = zip(grads, trainables)

    summaries = [tf.summary.histogram(var.op.name, var) for var in trainables]
    summaries += [tf.summary.histogram(g.op.name, g) for g in grads if g is not None]

    decayed_lr = tf.train.exponential_decay(params.learning_rate,
                                            tf.train.get_global_step(), 82000,
                                            0.8, staircase=True)
    optimizer = tf.train.AdamOptimizer(decayed_lr)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.apply_gradients(grad_var_pairs,
            global_step=tf.train.get_global_step())

    logging_hook = tf.train.LoggingTensorHook(
        {"Training loss": loss}, every_n_iter=10)

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      train_op=train_op,
                                      training_hooks=[logging_hook]
                                      )


hparams = tf.contrib.training.HParams(
    link_state_dim=4,
    path_state_dim=2,
    T=3,
    readout_units=8,
    learning_rate=0.001,
    batch_size=32,
    dropout_rate=0.5,
    l2=0.1,
    l2_2=0.01,
    learn_embedding=True # If false, only the readout is trained
)


def train(args):
    print(args)
    tf.logging.set_verbosity('INFO')

    if args.hparams:
        hparams.parse(args.hparams)

    my_checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=60*60,
        keep_checkpoint_max=20
    )

    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir=args.model_dir,
        params=hparams,
        warm_start_from=args.warm,
        config=my_checkpointing_config
        )

    train_spec = tf.estimator.TrainSpec(input_fn=lambda:tfrecord_input_fn(args.train,hparams,shuffle_buf=args.shuffle_buf,target=args.target),max_steps=args.train_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:tfrecord_input_fn(args.eval_,hparams,shuffle_buf=None,target=args.target),throttle_secs=5*60, steps=300)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

def extract_links(n, connections, link_cap):
    A = np.zeros((n,n))

    for a,c in zip(A,connections):
        a[c]=1

    G=nx.from_numpy_array(A, create_using=nx.DiGraph())
    edges=list(G.edges)
    capacities_links = []
    # The edges 0-2 or 2-0 can exist. They are duplicated (up and down) and they must have same capacity.
    for e in edges:
        if str(e[0])+':'+str(e[1]) in link_cap:
            capacity = link_cap[str(e[0])+':'+str(e[1])]
            capacities_links.append(capacity)
        elif str(e[1])+':'+str(e[0]) in link_cap:
            capacity = link_cap[str(e[1])+':'+str(e[0])]
            capacities_links.append(capacity)
        else:
            print("ERROR IN THE DATASET!")
            exit()
    return edges, capacities_links

def make_paths(R,connections, link_cap):
    n = R.shape[0]
    edges, capacities_links = extract_links(n, connections, link_cap)
    # link paths
    link_paths = []
    for i in range(n):
        for j in range(n):
            if i != j:
                link_paths.append([edges.index(tup) for tup in pairwise(genPath(R,i,j,connections))])
    # node paths
    node_paths = []
    for i in range(n):
        for j in range(n):
            if i != j:
                # last node is not relevant for node paths
                node_paths.append(list(genPath(R,i,j,connections))[:-1])

    return link_paths, node_paths, capacities_links

class NewParser:
    netSize = 0
    offsetDelay = 0
    hasPacketGen = True

    def __init__(self,netSize):
        self.netSize = netSize
        self.offsetDelay = netSize*netSize*3

    def getBwPtr(self,src,dst):
        return ((src*self.netSize + dst)*3)
    def getGenPcktPtr(self,src,dst):
        return ((src*self.netSize + dst)*3 + 1)
    def getDropPcktPtr(self,src,dst):
        return ((src*self.netSize + dst)*3 + 2)
    def getDelayPtr(self,src,dst):
        return (self.offsetDelay + (src*self.netSize + dst)*8)
    def getJitterPtr(self,src,dst):
        return (self.offsetDelay + (src*self.netSize + dst)*8 + 7)

def ned2lists(f):
    channels = []
    link_cap = {}
    queue_sizes_dict = {}

    node_id_regex = re.compile(r'\s+node(\d+): Server {')
    queue_size_regex = re.compile(r'\s+queueSize = (\d+);')
    current_node_id = None
    for line in f:
        line = line.decode()
        m1 = node_id_regex.match(line)
        if m1:
            current_node_id = int(m1.groups()[0])
        m2 = queue_size_regex.match(line)
        if m2:
            queue_sizes_dict[current_node_id] = int(m2.groups()[0])

    queue_sizes = [queue_sizes_dict[key] for key in sorted(queue_sizes_dict.keys())]
    f.seek(0)

    p = re.compile(r'\s+node(\d+).port\[(\d+)\]\s+<-->\s+Channel(\d+)kbps+\s<-->\s+node(\d+).port\[(\d+)\]')
    for line in f:
        line = line.decode()
        m=p.match(line)
        if m:
            auxList = []
            it = 0
            for elem in list(map(int,m.groups())):
                if it!=2:
                    auxList.append(elem)
                it = it + 1
            channels.append(auxList)
            link_cap[(m.groups()[0])+':'+str(m.groups()[3])] = int(m.groups()[2])

    n=max(map(max, channels))+1
    connections = [{} for i in range(n)]
    # Shape of connections[node][port] = node connected to
    for c in channels:
        connections[c[0]][c[1]]=c[2]
        connections[c[2]][c[3]]=c[0]
    # Connections store an array of nodes where each node position correspond to
    # another array of nodes that are connected to the current node
    connections = [[v for k,v in sorted(con.items())]
                   for con in connections ]
    return connections, n, link_cap, queue_sizes

def get_corresponding_values(posParser, line, n, bws, delays, jitters):
    bws.fill(0)
    delays.fill(0)
    jitters.fill(0)
    it = 0
    for i in range(n):
        for j in range(n):
            if i!=j:
                delay = posParser.getDelayPtr(i, j)
                jitter = posParser.getJitterPtr(i, j)
                traffic = posParser.getBwPtr(i, j)
                bws[it] = float(line[traffic])
                delays[it] = float(line[delay])
                jitters[it] = float(line[jitter])
                it = it + 1

def make_tfrecord2(directory, tf_file, ned_file, routing_file, data_file):
    con,n,link_cap, queue_sizes = ned2lists(ned_file)
    posParser = NewParser(n)

    R = load_routing(routing_file)
    link_paths, node_paths, link_capacities = make_paths(R, con, link_cap)

    n_link_paths = len(link_paths)
    n_node_paths = len(node_paths)
    assert n_link_paths == n_node_paths
    n_paths = n_link_paths

    n_links = max(max(link_paths)) + 1
    n_nodes = max(max(node_paths)) + 1


    a = np.zeros(n_paths)
    d = np.zeros(n_paths)
    j = np.zeros(n_paths)

    tfrecords_dir = directory+"tfrecords/"

    if not os.path.exists(tfrecords_dir):
        os.makedirs(tfrecords_dir)

    link_indices, link_path_indices, link_sequ_indices = make_indices(link_paths)
    node_indices, node_path_indices, node_sequ_indices = make_indices(node_paths)

    n_link_total = len(link_path_indices)
    n_node_total = len(node_path_indices)

    writer = tf.python_io.TFRecordWriter(tfrecords_dir + tf_file)

    for line in data_file:
        line = line.decode().split(',')
        get_corresponding_values(posParser, line, n, a, d, j)

        example = tf.train.Example(features=tf.train.Features(feature={
            'traffic':_float_features(a),
            'delay':_float_features(d),
            'jitter':_float_features(j),
            'link_capacity': _float_features(link_capacities),
            'queue_sizes': _float_features(queue_sizes),
            'links':_int64_features(link_indices),
            'link_paths':_int64_features(link_path_indices),
            'link_sequences':_int64_features(link_sequ_indices),
            'nodes':_int64_features(node_indices),
            'node_paths':_int64_features(node_path_indices),
            'node_sequences':_int64_features(node_sequ_indices),
            'n_links':_int64_feature(n_links),
            'n_nodes':_int64_feature(n_nodes),
            'n_paths':_int64_feature(n_paths),
            'n_link_total':_int64_feature(n_link_total),
            'n_node_total':_int64_feature(n_link_total)
        }
        ))

        writer.write(example.SerializeToString())
    writer.close()

def data(args):
    directory = args.d[0]
    nodes_dir = directory.split('/')[-1]
    if (nodes_dir==''):
        nodes_dir=directory.split('/')[-2]

    ned_filename = ""
    if nodes_dir=="nsfnetQueue":
        ned_filename = "/Network_nsfnetQueue.ned"
    elif nodes_dir=="geant2bwQueue":
        ned_filename = "/Network_geant2bw.ned"

    for filename in os.listdir(directory):
        if filename.endswith(".tar.gz"):
            print(filename)
            tf_file = filename.split('.')[0]+".tfrecords"
            tar = tarfile.open(directory+filename, "r:gz")

            dir_info = tar.next()
            if (not dir_info.isdir()):
                print("Tar file with wrong format")
                exit()

            #delay_file = tar.extractfile(dir_info.name + "/simulationResults.txt")
            delay_file = tar.extractfile(dir_info.name + "/delayGlobal.txt")
            routing_file = tar.extractfile(dir_info.name + "/Routing.txt")
            ned_file = tar.extractfile(dir_info.name + ned_filename)

            tf.logging.info('Starting ', delay_file)
            make_tfrecord2(directory, tf_file,ned_file,routing_file,delay_file)

    directory_tfr = directory+"tfrecords/"

    tfr_train = directory_tfr+"train/"
    tfr_eval = directory_tfr+"evaluate/"
    if not os.path.exists(tfr_train):
        os.makedirs(tfr_train)

    if not os.path.exists(tfr_eval):
        os.makedirs(tfr_eval)

    tfrecords = glob.glob(directory_tfr+ '*.tfrecords')
    training = len(tfrecords) * 0.8
    train_samples = random.sample(tfrecords, int(training))
    evaluate_samples = list(set(tfrecords) - set(train_samples))

    for file in train_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_train + file_name)

    for file in evaluate_samples:
        file_name = file.split('/')[-1]
        os.rename(file, tfr_eval + file_name)

def predict(args):
    tf.logging.set_verbosity('INFO')

    if args.hparams:
        hparams.parse(args.hparams)

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params=hparams
    )

    print(estimator.evaluate(lambda: tfrecord_input_fn(args.predict, hparams, shuffle_buf=None, target=args.target), steps=args.eval_steps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RouteNet: a Graph Neural Network model for computer network modeling')

    subparsers = parser.add_subparsers(help='sub-command help')
    parser_data = subparsers.add_parser('data', help='data processing')
    parser_data.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    parser_data.set_defaults(func=data)

    parser_train = subparsers.add_parser('train', help='Train options')
    parser_train.add_argument('--hparams', type=str,help='Comma separated list of "name=value" pairs.')
    parser_train.add_argument('--train', help='Train Tfrecords files', type=str, nargs='+')
    parser_train.add_argument('--eval_', help='Evaluation Tfrecords files', type=str, nargs='+')
    parser_train.add_argument('--model_dir', help='Model directory', type=str)
    parser_train.add_argument('--train_steps', help='Training steps', type=int, default=100)
    parser_train.add_argument('--eval_steps', help='Evaluation steps, defaul None= all', type=int, default=None)
    parser_train.add_argument('--shuffle_buf', help="Buffer size for samples shuffling", type=int, default=10000)
    parser_train.add_argument('--target', help="Predicted variable", type=str, default='delay')
    parser_train.add_argument('--warm',help = "Warm start from", type=str, default=None)
    parser_train.set_defaults(func=train)
    parser_train.set_defaults(name="Train")

    args = parser.parse_args()
    args.func(args)
