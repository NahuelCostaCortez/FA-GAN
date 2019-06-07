'''
Nahuel.- Evaluacion de los modelos 
'''

import tensorflow as tf
import numpy as np

def get_batch(samples, batch_size, batch_idx, labels=None):
    start_pos = batch_idx * batch_size
    end_pos = start_pos + batch_size
    return samples[start_pos:end_pos]

def normalise_data(train, valores_reales, low=-1, high=1):
    min_val = np.nanmin(train)
    max_val = np.nanmax(train)

    normalised_data = (valores_reales - min_val)/(max_val - min_val)
    normalised_data = (high - low)*normalised_data + low

    return normalised_data

limite = 0.5

# Hacemos que la salida sea 0 o 1 
def evaluation(output):
  final_output = []
  for i in range(len(output)):
    if np.mean(output[i]) > limite_media:
      final_output.append(1)
    else:
      final_output.append(0)
  return final_output

batch_size = 28

data_path = './data/markov_data999na180.npy'
samples = np.load(data_path).item()
samples = samples['samples']
train = samples['train']
#print('La longitud del conjunto de train es: ', len(train))
test = samples['test']
#print('La longitud del conjunto de test es: ', len(test))

sess=tf.Session()    
saver = tf.train.import_meta_graph('./trained_models/model-19.meta')
saver.restore(sess,('./trained_models/model-19'))

graph = tf.get_default_graph()
X = graph.get_tensor_by_name("Placeholder_1:0")
D_real = graph.get_tensor_by_name("discriminator_2/Sigmoid:0")

# Evaluacion reales 
'''
data_path = '../../datos/real_data.npy'
data = np.load(data_path).item()
real_values = data['samples']['real_values']
eval_limit = data['samples']['eval_limit']
patients = data['samples']['patient_list']

X_mb = normalise_data(train, real_values)
output = sess.run(D_real,feed_dict={X: X_mb})
prediction = evaluation2(output, eval_limit)
print("output 0:", output[0])
print("output 1:", output[1])
print("output 7:", output[7])
for i in range(len(prediction)):
	print('Prediction for patient ',patients[i],': ',prediction[i])
'''
#--------------------------------

accuracy = 0 
media = []
# Evaluar modelo con datos de TRAIN
for i in range(int(len(train)/ batch_size)):
	X_mb = get_batch(train, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.ones_like(output) 
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with train data: ', accuracy/(len(train)/ batch_size))
#--------------------------------

accuracy=0
# Evaluar modelo con los datos de TEST 
for i in range(int(len(test)/ batch_size)):
	X_mb = get_batch(test, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.ones_like(output) 
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with test data: ', accuracy/(len(test)/ batch_size))
#--------------------------------

# Evaluar el modelo con los datos SMOOTH 
data_path = './data/data_smooth.npy' 
samples = np.load(data_path).item() 
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
	X_mb = get_batch(samples, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.zeros_like(output)  
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with smooth data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con alpha=0.994
data_path = '../../datos/markov_data994.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
	X_mb = get_batch(samples, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.zeros_like(output)  
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with 994 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con alpha=0.997
data_path = '../../datos/markov_data_dias997.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
	X_mb = get_batch(samples, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.zeros_like(output)  
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with 997 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con alpha=0.998
data_path = '../../datos/markov_data_dias998.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
	X_mb = get_batch(samples, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.zeros_like(output)  
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with 998 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con ga15na145
data_path = '../../datos/markov_dataga15na145.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
	X_mb = get_batch(samples, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.zeros_like(output)  
	#print("Accuracy at ",i)
	#print(np.mean(np.rint(np.equal(labels, prediction)))*100)
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with ga15na145 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con na5
data_path = '../../datos/markov_datana5.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
        X_mb = get_batch(samples, batch_size, i)
        output = sess.run(D_real,feed_dict={X: X_mb})
        prediction = evaluation(output)
        #print("prediction at i: ", i)
        #print(output[0])
        labels = np.zeros_like(output)
        accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with na5 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con na30
data_path = '../../datos/markov_datana30.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
	X_mb = get_batch(samples, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.zeros_like(output)  
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with na30 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con na90
data_path = '../../datos/markov_datana90.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
        X_mb = get_batch(samples, batch_size, i)
        output = sess.run(D_real,feed_dict={X: X_mb})
        prediction = evaluation(output)
        labels = np.zeros_like(output)
        accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with na90 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con na120
data_path = '../../datos/markov_datana120.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
        X_mb = get_batch(samples, batch_size, i)
        output = sess.run(D_real,feed_dict={X: X_mb})
        prediction = evaluation(output)
        labels = np.zeros_like(output)
        accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with na120 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con na260
data_path = '../../datos/markov_datana260.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)): 
	X_mb = get_batch(samples, batch_size, i)
	output = sess.run(D_real,feed_dict={X: X_mb})
	prediction = evaluation(output)
	labels = np.zeros_like(output)  
	accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with na260 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------


# Evaluar el modelo con na300
data_path = '../../datos/markov_datana300.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
        X_mb = get_batch(samples, batch_size, i)
        output = sess.run(D_real,feed_dict={X: X_mb})
        prediction = evaluation(output)
        labels = np.zeros_like(output)
        accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with na300 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con a04
data_path = '../../datos/markov_data_diasa04.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
        X_mb = get_batch(samples, batch_size, i)
        output = sess.run(D_real,feed_dict={X: X_mb})
        prediction = evaluation(output)
        labels = np.zeros_like(output)
        accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with a04 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------

# Evaluar el modelo con a06
data_path = '../../datos/markov_data_diasa06.npy'
samples = np.load(data_path).item()
samples = samples['samples']['train']

accuracy=0
for i in range(int(len(samples)/ batch_size)):
        X_mb = get_batch(samples, batch_size, i)
        output = sess.run(D_real,feed_dict={X: X_mb})
        prediction = evaluation(output)
        labels = np.zeros_like(output)
        accuracy += np.mean(np.rint(np.equal(labels, prediction)))*100
print('Accuracy with a06 data: ', accuracy/(len(samples)/ batch_size))
#--------------------------------
