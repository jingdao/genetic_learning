import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import numpy
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
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

features_train = x_train.reshape(len(x_train),-1)
norm = features_train.sum(axis=1)
features_train /= norm.reshape(-1,1)

kmeans_label = KMeans(n_clusters=10, random_state=0).fit_predict(features_train)
nmi = normalized_mutual_info_score(y_train, kmeans_label)
ami = adjusted_mutual_info_score(y_train, kmeans_label)
ars = adjusted_rand_score(y_train, kmeans_label)
print("KMeans NMI: %.3f AMI: %.3f ARS: %.3f %d/%d clusters"% (nmi,ami,ars,len(numpy.unique(kmeans_label)),len(numpy.unique(y_train))))
spectral_label = SpectralClustering(n_clusters=10, random_state=0).fit_predict(features_train)
nmi = normalized_mutual_info_score(y_train, spectral_label)
ami = adjusted_mutual_info_score(y_train, spectral_label)
ars = adjusted_rand_score(y_train, spectral_label)
print("Spectral NMI: %.3f AMI: %.3f ARS: %.3f %d/%d clusters"% (nmi,ami,ars,len(numpy.unique(spectral_label)),len(numpy.unique(y_train))))

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
config.log_device_placement = False
N = 100 #number of learners
L = 600 #images per episode
cycles = len(x_train) / L
repeat_episodes = 10
M = int(L * 0.1) #batch size of learner
K = 10 #number of neighbors
nets = [None] * N
genes = [None] * N
tf.set_random_seed(0)
numpy.random.seed(0)

#initialize network weights
t = time.time()
for i in range(N):
	gene = options[range(options.shape[0]), numpy.random.randint(0,options.shape[1],options.shape[0])]
	gene = list(gene)[:5] + [2]
	net = MyNet(M,x_train.shape[-1],gene,i) 
	nets[i] = net
	genes[i] = gene
	sys.stdout.write('creating %d/%d nets\r'%(i+1,N))
	sys.stdout.flush()
sess = tf.Session(config=config)
init = tf.global_variables_initializer()
sess.run(init)
print("Network initialization: %.2fs"%(time.time() - t))

for e in range(10):
	episode = e / repeat_episodes
	train_images = x_train[(episode%cycles)*L:((episode%cycles)+1)*L, :, :]
	picks = []
	unpicks = []
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
		unpicks.append(set(bottom))
		top_labels = y_train[top]
		i,c = numpy.unique(top_labels,return_counts=True)
		mode.append(i[numpy.argmax(c)])
		inliers1 = 1.0 * numpy.max(c) / M
		inlier_acc.append(inliers1)
	print("Inlier computation: %.2fs"%(time.time() - t))

	pairwise_score = numpy.zeros((N, N))
	for i in range(N):
		for j in range(N):
			if i != j:
				score = len(picks[i].intersection(picks[j]))
				pairwise_score[i][j] = score
				pairwise_score[j][i] = score
	
	for i in range(N):
		scores = pairwise_score[i,:]
		c = numpy.argsort(scores)[::-1][:K]
		intersect_ratio.extend(scores[c] / M)
		positives = reduce(lambda x,y: x+y, [list(picks[j]) for j in c])
		negatives = reduce(lambda x,y: x+y, [list(unpicks[j]) for j in c])
		k,c = numpy.unique(y_train[positives], return_counts=True)
		m = k[numpy.argmax(c)]
		pr = 1.0 * numpy.sum(y_train[positives]==m) / len(positives)
		nr = 1.0 * numpy.sum(y_train[negatives]!=m) / len(negatives)
		intersect_acc.append(pr)
		intersect_acc.append(nr)

		train_labels = numpy.array([1] * len(positives) + [0] * len(negatives))
		train_idx = numpy.array(positives + negatives)
		shuffled_idx = numpy.arange(len(train_labels))
		numpy.random.shuffle(shuffled_idx)
		num_batches = len(train_labels) / M
		for j in range(num_batches):
			loss,_ = sess.run([nets[i].loss,nets[i].train_op], feed_dict={
				nets[i].images_pl: train_images[train_idx[shuffled_idx[j*M:(j+1)*M]]],
				nets[i].labels_pl: train_labels[shuffled_idx[j*M:(j+1)*M]]
			})
			losses.append(loss)

	print('Episode %d: loss: %.2f inlier: %.2f intersect %.2f %.2f'%(episode,numpy.mean(losses),numpy.mean(inlier_acc),numpy.mean(intersect_ratio),numpy.mean(intersect_acc)))

affinity = numpy.zeros((len(train_images), len(train_images)))
for i in range(N):
	probs = []
	for j in range(len(train_images) / M):
		start_idx = j * M
		end_idx = (j+1) * M
		p = sess.run(nets[i].probs, feed_dict={nets[i].images_pl:train_images[start_idx:end_idx]})[:,1]
		probs.extend(p)
#	order = numpy.argsort(probs)
#	top = order[-M:]
	top = numpy.nonzero(numpy.array(probs) > 0.5)[0]
	for u,v in itertools.combinations(sorted(top),2):
		affinity[u,v] += 1
		affinity[v,u] += 1

#plt.imshow(affinity)
#plt.show()

spectral_label = SpectralClustering(n_clusters=10, random_state=0, affinity='precomputed').fit_predict(affinity)
nmi = normalized_mutual_info_score(y_train, spectral_label)
ami = adjusted_mutual_info_score(y_train, spectral_label)
ars = adjusted_rand_score(y_train, spectral_label)
print("Spectral NMI: %.3f AMI: %.3f ARS: %.3f %d/%d clusters"% (nmi,ami,ars,len(numpy.unique(spectral_label)),len(numpy.unique(y_train))))
