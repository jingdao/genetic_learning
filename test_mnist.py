import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import numpy
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import sys
import time
import networkx as nx
import itertools
import matplotlib.pyplot as plt

(x_train, y_train),(x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
ratio = 0.01
num_train = int(ratio * len(y_train))
num_test = int(ratio * len(y_test))
x_train = x_train[:num_train, :, :]
y_train = y_train[:num_train]
x_test = x_test[:num_test, :, :]
y_test = y_test[:num_test]
print('train',x_train.shape,y_train.shape)
print('test',x_test.shape,y_test.shape)

#Gene
#window1 {3,4,5}
#channels1 {5,10,15}
#window2 {3,4,5}
#channels2 {10,20,30}
#channels3 {50,100,150}
#channels4 {10,20,50}
class MyNet():
	def __init__(self,batch_size, img_size, gene, net_id):
		self.images_pl = tf.placeholder(tf.float32, shape=(batch_size, img_size, img_size))
		self.labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
		window1, channels1, window2, channels2, channels3, num_class = gene

		self.conv1 = tf.reshape(self.images_pl, [batch_size, img_size, img_size, 1])
		self.kernel1 = tf.get_variable('net'+str(net_id)+'kernel1', [window1,window1,1,channels1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias1 = tf.get_variable('net'+str(net_id)+'bias1', [channels1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.conv1 = tf.nn.conv2d(self.conv1, self.kernel1, [1,1,1,1], padding='VALID')
		self.conv1 = tf.nn.bias_add(self.conv1, self.bias1)
		self.conv1 = tf.nn.relu(self.conv1)
		self.pool1 = tf.nn.max_pool(self.conv1,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='net'+str(net_id)+'pool1')

		self.kernel2 = tf.get_variable('net'+str(net_id)+'kernel2', [window2,window2,channels1,channels2], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias2 = tf.get_variable('net'+str(net_id)+'bias2', [channels2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.conv2 = tf.nn.conv2d(self.pool1, self.kernel2, [1,1,1,1], padding='VALID')
		self.conv2 = tf.nn.bias_add(self.conv2, self.bias2)
		self.conv2 = tf.nn.relu(self.conv2)
		self.pool2 = tf.nn.max_pool(self.conv2,ksize=[1,2,2,1],strides=[1,2,2,1], padding='VALID', name='net'+str(net_id)+'pool2')

		prev_channels = int(numpy.prod(self.pool2.shape[1:]))
		self.fc3 = tf.reshape(self.pool2, [batch_size, prev_channels])
		self.kernel3 = tf.get_variable('net'+str(net_id)+'kernel3', [prev_channels,channels3], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias3 = tf.get_variable('net'+str(net_id)+'bias3', [channels3], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc3 = tf.matmul(self.fc3, self.kernel3)
		self.fc3 = tf.nn.bias_add(self.fc3, self.bias3)
		self.fc3 = tf.nn.relu(self.fc3)

		self.kernel4 = tf.get_variable('net'+str(net_id)+'kernel4', [channels3, num_class], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
		self.bias4 = tf.get_variable('net'+str(net_id)+'bias4', [num_class], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
		self.fc4 = tf.matmul(self.fc3, self.kernel4)
		self.output = tf.nn.bias_add(self.fc4, self.bias4)
		self.probs = tf.nn.softmax(self.output, dim=1)

		#loss functions
		self.class_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.labels_pl))
		self.loss = self.class_loss
		self.correct = tf.equal(tf.argmax(self.output, 1), tf.to_int64(self.labels_pl))
		self.accuracy = tf.reduce_sum(tf.cast(self.correct, tf.float32)) / float(batch_size)

		#optimizer
		self.batch = tf.Variable(0)
		self.learning_rate = tf.train.exponential_decay(0.001,self.batch,122000,0.5,staircase=True)
		self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op = self.optimizer.minimize(self.loss, global_step=self.batch)

options = numpy.array([
	[3,4,5],
	[5,10,15],
	[3,4,5],
	[10,20,30],
	[50,100,150],
	[10,20,50]
])

if True:
#	features_train = x_train.reshape(len(x_train),-1)
#	norm = features_train.sum(axis=1)
#	features_train /= norm.reshape(-1,1)
#
#	kmeans_label = KMeans(n_clusters=10, random_state=0).fit_predict(features_train)
#	nmi = normalized_mutual_info_score(y_train, kmeans_label)
#	ami = adjusted_mutual_info_score(y_train, kmeans_label)
#	ars = adjusted_rand_score(y_train, kmeans_label)
#	print("NMI: %.3f AMI: %.3f ARS: %.3f %d/%d clusters"% (nmi,ami,ars,len(numpy.unique(kmeans_label)),len(numpy.unique(y_train))))

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	config.log_device_placement = False
	N = 100 #number of learners
	L = 600 #images per episode
	cycles = len(x_train) / L
	repeat_episodes = 10
	M = int(L * 0.05) #batch size of learner
	nets = [None] * N
	genes = [None] * N
	tf.set_random_seed(0)
	numpy.random.seed(0)

	#initialize network weights
	t = time.time()
	for i in range(N):
		gene = options[range(options.shape[0]), numpy.random.randint(0,options.shape[1],options.shape[0])]
		gene = list(gene)[:5] + [2]
#		gene = [5,10,5,20,100,2]
		net = MyNet(M,x_train.shape[-1],gene,i) 
		nets[i] = net
		genes[i] = gene
		sys.stdout.write('creating %d/%d nets\r'%(i+1,N))
		sys.stdout.flush()
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)
	print("Network initialization: %.2fs"%(time.time() - t))

	for e in range(30):
		episode = e / repeat_episodes
		train_images = x_train[(episode%cycles)*L:((episode%cycles)+1)*L, :, :]
		picks = []
		mode = []
		inlier_acc = []
		intersect_ratio = []
		intersect_acc = []
		losses = []
		t = time.time()
		for i in range(N):
			probs = []
			for j in range(len(train_images) / M):
				start_idx = j * M
				end_idx = (j+1) * M
				p = sess.run(nets[i].probs, feed_dict={nets[i].images_pl:train_images[start_idx:end_idx]})[:,1]
				probs.extend(p)
			order = numpy.argsort(probs)
			top = order[-M:]
			bottom = order[:M]
			picks.append(set(top))
			picks.append(set(bottom))
			top_labels = y_train[top]
			i,c = numpy.unique(top_labels,return_counts=True)
			mode.append(i[numpy.argmax(c)])
			inliers1 = 1.0 * numpy.max(c) / M
			bottom_labels = y_train[bottom]
			i,c = numpy.unique(bottom_labels,return_counts=True)
			mode.append(i[numpy.argmax(c)])
			inliers2 = 1.0 * numpy.max(c) / M
			inlier_acc.append(max(inliers1, inliers2))
#			print(list(top_labels) + list(bottom_labels))
#		print("Inlier computation: %.2fs"%(time.time() - t))

		G = nx.Graph()
		validated = set()
		for p in picks:
			edges = list(itertools.combinations(sorted(list(p)), 2))
			for e in validated.intersection(edges):
				if e in G.edges:
					w = G.edges[e]['weight']
					G.add_edge(e[0],e[1],weight=w+1)
				else:
					G.add_edge(e[0],e[1],weight=1)
			validated.update(edges)
		print('Created graph with %d nodes %d/%d edges'%(len(G.nodes), len(G.edges), len(validated)))
		weights = [d['weight'] for (u,v,d) in G.edges.data()]
		print('Weights %d %.2f %d'%(numpy.min(weights),numpy.mean(weights),numpy.max(weights)))
		nx.draw(G, node_color=y_train[list(G.nodes)], labels={i:y_train[i] for i in G.nodes}, width=weights)
		plt.show()
		sys.exit(1)

		t = time.time()
		for i in range(N):
			best_pair = (None,None,None,None)
			max_intersect = 0
			for j in range(2*N):
				if 2*i!=j:
					num_intersect = len(picks[2*i].intersection(picks[j]))
					if num_intersect > max_intersect:
						max_intersect = num_intersect
						best_pair = (i,j/2,0,j%2)
				if 2*i+1!=j:
					num_intersect = len(picks[2*i+1].intersection(picks[j]))
					if num_intersect > max_intersect:
						max_intersect = num_intersect
						best_pair = (i,j/2,1,j%2)
			intersect_ratio.append(1.0 * max_intersect / M)
			intersect_acc.append(mode[2*best_pair[0]+best_pair[2]] == mode[2*best_pair[1]+best_pair[3]])
			print('%d/%d intersects (%d<->%d) %s%s'%(max_intersect,M, mode[2*best_pair[0]+best_pair[2]], mode[2*best_pair[1]+best_pair[3]], '-' if best_pair[2] else '+', '-' if best_pair[3] else '+'))

			train_idx = list(picks[2*i + best_pair[2]])
			lb = [k in picks[2*best_pair[1] + best_pair[3]] for k in train_idx]
			if best_pair[2]: #flip sign if bottom inliers
				lb = numpy.logical_not(lb)
			loss,_ = sess.run([nets[i].loss,nets[i].train_op], feed_dict={nets[i].images_pl:train_images[train_idx], nets[i].labels_pl:lb})
			losses.append(loss)
#		print("Network training: %.2fs"%(time.time() - t))
		print('Episode %d: loss: %.2f inlier: %.2f intersect %.2f %.2f'%(episode,numpy.mean(losses),numpy.mean(inlier_acc), numpy.mean(intersect_ratio), numpy.mean(intersect_acc)))

	sys.exit(1)

BATCH_SIZE = 100
VAL_STEP = 50
MAX_EPOCH = 50

with tf.Graph().as_default():
	with tf.device('/gpu:0'):
		gene = options[range(options.shape[0]), numpy.random.randint(0,options.shape[1],options.shape[0])]
#		gene = [5,10,5,20,100,10]
		print(gene)
		net = MyNet(BATCH_SIZE,x_train.shape[-1],list(gene)[:-1] + [y_train.max()+1]) 
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True
		config.log_device_placement = False
		sess = tf.Session(config=config)
		init = tf.global_variables_initializer()
		sess.run(init)

		for epoch in range(MAX_EPOCH):
			#shuffle data
			idx = numpy.arange(len(y_train))
			numpy.random.shuffle(idx)
			shuffled_images = x_train[idx, :, :]
			shuffled_labels = y_train[idx]

			#split into batches
			num_batches = int(len(y_train) / BATCH_SIZE)
			losses = []
			accuracies = []
			for batch_id in range(num_batches):
				start_idx = batch_id * BATCH_SIZE
				end_idx = (batch_id + 1) * BATCH_SIZE
				feed_dict = {net.images_pl: shuffled_images[start_idx:end_idx,:,:],
					net.labels_pl: shuffled_labels[start_idx:end_idx]}
				loss,accuracy,_ = sess.run([net.loss,net.accuracy,net.train_op],feed_dict=feed_dict)
				losses.append(loss)
				accuracies.append(accuracy)
#			print('Epoch: %d Loss: %.3f Accuracy: %.3f'%(epoch, numpy.mean(losses),numpy.mean(accuracies)))

			if epoch % VAL_STEP == VAL_STEP - 1:
				#get validation loss
				num_batches = int(len(y_test) / BATCH_SIZE)
				losses = []
				accuracies = []
				for batch_id in range(num_batches):
					start_idx = batch_id * BATCH_SIZE
					end_idx = (batch_id + 1) * BATCH_SIZE
					feed_dict = {net.images_pl: x_test[start_idx:end_idx,:,:],
						net.labels_pl: y_test[start_idx:end_idx]}
					loss,accuracy = sess.run([net.loss,net.accuracy], feed_dict=feed_dict)
					losses.append(loss)
					accuracies.append(accuracy)
				print('Validation: %d Loss: %.3f Accuracy: %.3f'%(epoch,numpy.mean(losses),numpy.mean(accuracies)))

#		saver.save(sess, MODEL_PATH)
